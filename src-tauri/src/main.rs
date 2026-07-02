use std::path::PathBuf;
use std::sync::Mutex;

use candle_core::Device;
use serde::Serialize;
use tauri::{Manager, State};
use tokenizers::Tokenizer;

mod model_wrapper;
use model_wrapper::QuantizedModel;

struct AppState {
    model: Mutex<Option<Box<dyn QuantizedModel + Send>>>,
    tokenizer: Mutex<Option<Tokenizer>>,
    load_error: Mutex<Option<String>>,
}

#[derive(Serialize, Clone)]
struct CodeError {
    line: usize,
    #[serde(rename = "type")]
    err_type: String,
    description: String,
    fix: String,
}

#[derive(Serialize)]
struct AnalysisResponse {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    errors: Option<Vec<CodeError>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

fn model_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("code-reviewer")
}

fn build_prompt(code: &str) -> String {
    let system = "You are a senior code reviewer. Analyze the user code and find bugs, \
        logic errors, and bad practices. Reply ONLY with valid JSON, no markdown. \
        Format: {\\\"errors\\\":[{\\\"line\\\":N,\\\"type\\\":\\\"TYPE\\\",\\\"description\\\":\\\"DESC\\\",\\\"fix\\\":\\\"FIX\\\"}]} \
        If no errors: {\\\"errors\\\":[]} \
        Possible types: logic_error, runtime_error, bad_practice, security, performance, type_error. \
        Keep descriptions short. Keep fixes as corrected code snippets.";
    format!(
        "<|im_start|>system\n{}\n<|im_start|>user\n{}\n<|im_start|>assistant\n",
        system, code
    )
}

fn extract_json_errors(raw: &str) -> Vec<CodeError> {
    let mut errors = Vec::new();
    let marker = r#""errors":"#;
    let Some(start) = raw.find(marker) else {
        return errors;
    };
    let arr_start = start + marker.len();
    let rest = &raw[arr_start..];
    let trimmed = rest.trim_start();
    if !trimmed.starts_with('[') {
        return errors;
    }
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape_next = false;
    let mut end_idx = 0;
    for (i, ch) in trimmed.char_indices() {
        if escape_next {
            escape_next = false;
            continue;
        }
        if ch == '\\' && in_string {
            escape_next = true;
            continue;
        }
        if ch == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        if ch == '[' {
            depth += 1;
        } else if ch == ']' {
            depth -= 1;
            if depth == 0 {
                end_idx = i + 1;
                break;
            }
        }
    }
    if end_idx == 0 {
        return errors;
    }
    let arr_str = &trimmed[..end_idx];
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(arr_str) {
        if let Some(arr) = val.as_array() {
            for item in arr {
                let line = item.get("line").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let err_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();
                let description = item.get("description").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let fix = item.get("fix").and_then(|v| v.as_str()).unwrap_or("").to_string();
                errors.push(CodeError { line, err_type, description, fix });
            }
        }
    }
    errors
}

#[tauri::command]
fn analyze_code(code: String, state: State<AppState>) -> AnalysisResponse {
    let load_err = state.load_error.lock().unwrap();
    if let Some(msg) = load_err.as_ref() {
        return AnalysisResponse { ok: false, errors: None, error: Some(msg.clone()) };
    }
    drop(load_err);

    let mut model_lock = state.model.lock().unwrap();
    let tokenizer_lock = state.tokenizer.lock().unwrap();

    let model = match model_lock.as_mut() {
        Some(m) => m.as_mut(),
        None => return AnalysisResponse { ok: false, errors: None, error: Some("Model not loaded".into()) },
    };
    let tokenizer = match tokenizer_lock.as_ref() {
        Some(t) => t,
        None => return AnalysisResponse { ok: false, errors: None, error: Some("Tokenizer not loaded".into()) },
    };

    let prompt = build_prompt(&code);
    let tokens = match tokenizer.encode(prompt, true) {
        Ok(enc) => enc.get_ids().to_vec(),
        Err(e) => return AnalysisResponse { ok: false, errors: None, error: Some(format!("Tokenization error: {e}")) },
    };

    let max_new_tokens: usize = 1024;
    let mut generated = tokens.clone();

    let mut gen_error: Option<candle_core::Error> = None;
    for _ in 0..max_new_tokens {
        let input_len = generated.len();
        let start = input_len.saturating_sub(2048);
        let input_tokens = &generated[start..];

        let logits = match model.forward(input_tokens, 0) {
            Ok(l) => l,
            Err(e) => { gen_error = Some(e); break; }
        };
        let next_token_logits = match logits
            .get(logits.dim(0).unwrap_or(0) - 1)
            .and_then(|last| last.get(last.dim(0).unwrap_or(0) - 1))
        {
            Ok(l) => l,
            Err(e) => { gen_error = Some(e); break; }
        };

        let next_token = match next_token_logits
            .argmax(candle_core::D::Minus1)
            .and_then(|t| t.to_scalar::<u32>())
        {
            Ok(t) => t,
            Err(e) => { gen_error = Some(e); break; }
        };

        generated.push(next_token);

        let eos = tokenizer.token_to_id("<eos>").unwrap_or(0);
        if next_token == eos {
            break;
        }
    }

    if let Some(e) = gen_error {
        return AnalysisResponse { ok: false, errors: None, error: Some(format!("Inference error: {e}")) };
    }

    let output = match tokenizer.decode(&generated[tokens.len()..], true) {
        Ok(s) => s,
        Err(e) => return AnalysisResponse { ok: false, errors: None, error: Some(format!("Decode error: {e}")) },
    };

    let errors = extract_json_errors(&output);
    AnalysisResponse { ok: true, errors: Some(errors), error: None }
}

fn main() {
    let _app = tauri::Builder::default()
        .manage(AppState {
            model: Mutex::new(None),
            tokenizer: Mutex::new(None),
            load_error: Mutex::new(None),
        })
        .setup(|app| {
            let dir = model_dir();
            let mp = dir.join("model.gguf");
            let tp = dir.join("tokenizer.json");

            if !mp.exists() || !tp.exists() {
                let msg = format!(
                    "Model files not found.\n\nPlace model.gguf and tokenizer.json in:\n{}\n\nDownload TinyLlama 1.1B Q4_K_M from HuggingFace.",
                    dir.display()
                );
                let state = app.state::<AppState>();
                *state.load_error.lock().unwrap() = Some(msg);
                eprintln!("Model not found, continuing without LLM.");
                return Ok(());
            }

            eprintln!("Loading model from {}...", mp.display());
            let device = Device::Cpu;
            let mut f = std::fs::File::open(&mp)
                .map_err(|e| format!("Cannot open model: {e}"))?;
            let content = candle_core::quantized::gguf_file::Content::read(&mut f)
                .map_err(|e| format!("GGUF parse error: {e}"))?;
            let model = candle_transformers::models::quantized_llama::ModelWeights::from_gguf(
                content, &mut f, &device,
            )
            .map_err(|e| format!("GGUF load error: {e}"))?;

            let tokenizer = Tokenizer::from_file(&tp)
                .map_err(|e| format!("Tokenizer load error: {e}"))?;

            let state = app.state::<AppState>();
            *state.model.lock().unwrap() = Some(Box::new(model));
            *state.tokenizer.lock().unwrap() = Some(tokenizer);
            eprintln!("Model loaded successfully!");
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![analyze_code])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
