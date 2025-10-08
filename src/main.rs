use anyhow::{Context, Result};
use dotenvy::dotenv;
use once_cell::sync::Lazy;
use regex::Regex;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::io::{self, Write};

const DEEPSEEK_ENDPOINT: &str = "https://api.deepseek.com/chat/completions";
const MODEL: &str = "deepseek-chat";
const DEBATE_ROUNDS: usize = 5; // number of debate rounds
const MAX_TOKENS_AB_MSG: u32 = 80; // token cap for A/B turns
const MAX_TOKENS_C_START: u32 = 800; // token cap for C kick-off
const MAX_TOKENS_C_ROUND: u32 = 900; // token cap per C round
const MAX_TOKENS_C_FINAL: u32 = 1200; // token cap for C finale
const TEMP_AB: f32 = 0.7; // temperature for A/B
const TEMP_C: f32 = 0.2; // temperature for C
const STATE_FILE: &str = "shortlang_state.json"; // persistent state file
const MAX_CONTEXT_SNIPPET: usize = 320; // max chars when embedding prior turns

#[derive(Serialize, Deserialize, Clone, Debug)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: ChatMessage,
}

#[derive(Debug, Clone, Default)]
struct LibrarianResponse {
    delta_g: String,
    audit: String,
    sum: String,
    decode: String,
    glossary: String,
    final_answer: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
struct PersistentState {
    glossary: String,
    delta_g: String,
    #[serde(default)]
    delta_g_history: Vec<String>,
}

#[allow(dead_code)]
async fn complete_chat(
    client: &Client,
    api_key: &str,
    messages: Vec<ChatMessage>,
) -> Result<String> {
    complete_chat_with_tokens(client, api_key, messages, None, None).await
}

async fn complete_chat_with_tokens(
    client: &Client,
    api_key: &str,
    messages: Vec<ChatMessage>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
) -> Result<String> {
    let request_body = ChatRequest {
        model: MODEL.to_string(),
        messages,
        max_tokens,
        stream: None,
        temperature,
    };

    let response = client
        .post(DEEPSEEK_ENDPOINT)
        .bearer_auth(api_key)
        .json(&request_body)
        .send()
        .await
        .context("DeepSeek request failed")?
        .error_for_status()
        .context("DeepSeek returned an error status")?
        .json::<ChatCompletionResponse>()
        .await
        .context("Could not parse DeepSeek response as JSON")?;

    let content = response
        .choices
        .into_iter()
        .next()
        .map(|c| c.message.content)
        .context("DeepSeek response contained no choices")?;

    Ok(content)
}

async fn stream_chat_with_tokens(
    client: &Client,
    api_key: &str,
    messages: Vec<ChatMessage>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
) -> Result<String> {
    use std::io::Write as _;

    let request_body = ChatRequest {
        model: MODEL.to_string(),
        messages,
        max_tokens,
        stream: Some(true),
        temperature,
    };

    let mut response = client
        .post(DEEPSEEK_ENDPOINT)
        .bearer_auth(api_key)
        .json(&request_body)
        .send()
        .await
        .context("DeepSeek streaming request failed")?
        .error_for_status()
        .context("DeepSeek returned an error status for streaming")?;

    let mut buffer = String::new();
    let mut accumulated = String::new();
    let mut done = false;

    while !done {
        if let Some(chunk) = response
            .chunk()
            .await
            .context("Failed to read streaming chunk")?
        {
            let part = String::from_utf8_lossy(&chunk);
            buffer.push_str(&part);

            loop {
                if let Some(pos) = buffer.find('\n') {
                    let mut line = buffer[..pos].to_string();
                    buffer.drain(..=pos);
                    if line.ends_with('\r') {
                        line.pop();
                    }
                    if line.is_empty() || line.starts_with(":") {
                        continue;
                    }
                    if let Some(rest) = line.strip_prefix("data: ") {
                        let trimmed = rest.trim();
                        if trimmed == "[DONE]" {
                            done = true;
                            break;
                        }
                        if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
                            if let Some(delta) = v
                                .get("choices")
                                .and_then(|c| c.get(0))
                                .and_then(|c| c.get("delta"))
                                .and_then(|d| d.get("content"))
                                .and_then(|c| c.as_str())
                            {
                                print!("{}", delta);
                                io::stdout().flush().ok();
                                accumulated.push_str(delta);
                            }
                        }
                    }
                } else {
                    break;
                }
            }
        } else {
            break;
        }
    }

    println!("");
    Ok(accumulated)
}

fn shorten_for_prompt(text: &str, max_chars: usize) -> String {
    let trimmed = text.trim();
    if trimmed.len() <= max_chars {
        return trimmed.to_string();
    }
    let mut shortened = String::with_capacity(max_chars + 3);
    let mut count = 0;
    for ch in trimmed.chars() {
        let ch_len = ch.len_utf8();
        if count + ch_len > max_chars {
            shortened.push_str(" ...");
            break;
        }
        shortened.push(ch);
        count += ch_len;
    }
    shortened
}

fn extract_json_slice(text: &str) -> Option<String> {
    let mut s = text.trim();
    if s.starts_with("```") {
        s = s.trim_start_matches('`');
        if let Some(pos) = s.find('\n') {
            s = &s[pos + 1..];
        }
    }
    if s.ends_with("```") {
        if let Some(pos) = s.rfind("```") {
            s = &s[..pos];
        }
    }

    let bytes = s.as_bytes();
    let mut depth: i32 = 0;
    let mut in_string = false;
    let mut escaped = false;
    let mut start_idx: Option<usize> = None;
    for (idx, &byte) in bytes.iter().enumerate() {
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }
            if byte == b'\\' {
                escaped = true;
                continue;
            }
            if byte == b'"' {
                in_string = false;
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'{' => {
                if depth == 0 {
                    start_idx = Some(idx);
                }
                depth += 1;
            }
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(start) = start_idx {
                        return Some(s[start..=idx].to_string());
                    }
                }
            }
            _ => {}
        }
    }
    None
}

fn value_to_compact_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(arr) => {
            let parts: Vec<String> = arr.iter().map(value_to_compact_string).collect();
            parts.join(" | ")
        }
        Value::Object(_) => serde_json::to_string(v).unwrap_or_default(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn pick_field<'a>(root: &'a Value, names: &[&str]) -> Option<&'a Value> {
    for name in names {
        if let Some(val) = root.get(*name) {
            return Some(val);
        }
    }
    None
}

fn parse_librarian_response(text: &str) -> Result<LibrarianResponse> {
    let slice = extract_json_slice(text).unwrap_or_else(|| text.to_string());
    let v: Value = serde_json::from_str(&slice).context("Librarian response is not valid JSON")?;

    const DELTA_KEYS: [&str; 4] = ["DELTA_G", "delta_g", "DeltaG", "DG"];
    const AUDIT_KEYS: [&str; 2] = ["AUDIT", "audit"];
    const SUM_KEYS: [&str; 2] = ["SUM", "sum"];
    const DECODE_KEYS: [&str; 3] = ["DECODE", "decode", "interpretation"];
    const GLOSSARY_KEYS: [&str; 3] = ["GLOSSARY", "glossary", "lexicon"];
    const FINAL_KEYS: [&str; 3] = ["FINAL_ANSWER", "final_answer", "final"];

    let delta = value_to_compact_string(pick_field(&v, &DELTA_KEYS).unwrap_or(&Value::Null));
    let audit = value_to_compact_string(pick_field(&v, &AUDIT_KEYS).unwrap_or(&Value::Null));
    let sum = value_to_compact_string(pick_field(&v, &SUM_KEYS).unwrap_or(&Value::Null));
    let decode = value_to_compact_string(pick_field(&v, &DECODE_KEYS).unwrap_or(&Value::Null));
    let glossary = value_to_compact_string(pick_field(&v, &GLOSSARY_KEYS).unwrap_or(&Value::Null));
    let final_answer = pick_field(&v, &FINAL_KEYS)
        .map(value_to_compact_string)
        .filter(|s| !s.trim().is_empty());

    Ok(LibrarianResponse {
        delta_g: delta,
        audit,
        sum,
        decode,
        glossary,
        final_answer,
    })
}

fn parse_glossary_to_map(glossary_text: &str) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    let normalized = glossary_text.replace("||", ";");
    for part in normalized.split(';') {
        let token = part.trim();
        if token.is_empty() {
            continue;
        }
        for sub in token.split('|') {
            let entry = sub.trim();
            if entry.is_empty() {
                continue;
            }
            if let Some((code, desc)) = entry.split_once(':') {
                let code = code.trim().to_string();
                let desc = desc.trim().to_string();
                if !code.is_empty() {
                    map.insert(code, desc);
                }
            } else {
                map.insert(entry.to_string(), String::new());
            }
        }
    }
    map
}

fn format_glossary_from_map(map: &BTreeMap<String, String>) -> String {
    map.iter()
        .map(|(code, desc)| {
            if desc.is_empty() {
                code.clone()
            } else {
                format!("{}: {}", code, desc)
            }
        })
        .collect::<Vec<_>>()
        .join("; ")
}

fn normalize_delta_ops(delta_text: &str) -> Vec<String> {
    let cleaned = delta_text
        .replace('\n', " ")
        .replace('\r', " ")
        .replace("||", ";");
    let mut ops = Vec::new();
    let mut current = String::new();

    let mut chars = cleaned.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '+' | '-' => {
                if !current.trim().is_empty() {
                    ops.push(current.trim().to_string());
                }
                current.clear();
                current.push(ch);
                while let Some(&next_ch) = chars.peek() {
                    if next_ch == '+' || next_ch == '-' {
                        break;
                    }
                    if next_ch == ';' || next_ch == '|' {
                        chars.next();
                        break;
                    }
                    current.push(next_ch);
                    chars.next();
                }
                ops.push(current.trim().to_string());
                current.clear();
            }
            ';' | '|' => {
                if !current.trim().is_empty() {
                    ops.push(current.trim().to_string());
                }
                current.clear();
            }
            _ => {
                current.push(ch);
            }
        }
    }
    if !current.trim().is_empty() {
        ops.push(current.trim().to_string());
    }
    ops.into_iter()
        .filter(|op| op.starts_with('+') || op.starts_with('-'))
        .collect()
}

fn reconcile_glossary(base_glossary: &str, delta_text: &str) -> String {
    let mut map = parse_glossary_to_map(base_glossary);
    for op in normalize_delta_ops(delta_text) {
        let trimmed = op.trim();
        if trimmed.is_empty() {
            continue;
        }
        let (sign, rest) = match trimmed.chars().next() {
            Some('+') => ('+', trimmed[1..].trim()),
            Some('-') => ('-', trimmed[1..].trim()),
            _ => continue,
        };
        match sign {
            '+' => {
                let (code, desc) = if let Some((c, d)) = rest.split_once(':') {
                    (c.trim(), d.trim())
                } else {
                    (rest, "")
                };
                if !code.is_empty() {
                    map.insert(code.to_string(), desc.to_string());
                }
            }
            '-' => {
                let code = rest
                    .split(|ch: char| ch.is_whitespace() || ch == ':' || ch == ';' || ch == '|')
                    .next()
                    .unwrap_or("")
                    .trim();
                if !code.is_empty() {
                    map.remove(code);
                }
            }
            _ => {}
        }
    }
    format_glossary_from_map(&map)
}

static CODE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\b([A-Z][A-Z0-9]{1,5})\b").expect("invalid code regex"));

fn extract_codes(msg: &str) -> Vec<String> {
    let mut codes: Vec<String> = CODE_RE
        .captures_iter(msg)
        .filter_map(|cap| cap.get(1).map(|m| m.as_str().to_string()))
        .collect();
    codes.sort();
    codes.dedup();
    codes
}

fn load_state() -> Option<PersistentState> {
    match fs::read_to_string(STATE_FILE) {
        Ok(s) => serde_json::from_str(&s).ok(),
        Err(_) => None,
    }
}

fn save_state(state: &PersistentState) -> Result<()> {
    let serialized =
        serde_json::to_string_pretty(state).context("Failed to serialize persistent state")?;
    fs::write(STATE_FILE, serialized).context("Failed to write persistent state")?;
    Ok(())
}

fn librarian_system_prompt() -> ChatMessage {
    ChatMessage {
        role: "system".to_string(),
        content: "You are C (Librarian). You gatekeep glossary codes. Accept at most two new codes per round when A/B propose them with 'DELTA+ CODE:desc'. Reject invalid codes using '-CODE'. Never invent codes on your own. Always reply with strict JSON containing keys DELTA_G, AUDIT, SUM, DECODE, GLOSSARY and optionally FINAL_ANSWER.".to_string(),
    }
}

fn builder_system_prompt() -> ChatMessage {
    ChatMessage {
        role: "system".to_string(),
        content: "You are A (Builder). Communicate in compact short-code style. Stay within ~35 tokens and reuse at least 70% of the existing glossary codes. You may propose 0-2 new codes via 'DELTA+ CODE:short-desc'.".to_string(),
    }
}

fn challenger_system_prompt() -> ChatMessage {
    ChatMessage {
        role: "system".to_string(),
        content: "You are B (Challenger). Communicate in compact short-code style. Stay within ~35 tokens and reuse at least 70% of the existing glossary codes. You may propose 0-2 new codes via 'DELTA+ CODE:short-desc'. Focus on testing assumptions and surfacing risks.".to_string(),
    }
}

fn recent_history_snippet(history: &[String], limit: usize) -> String {
    history
        .iter()
        .rev()
        .take(limit)
        .filter(|entry| !entry.trim().is_empty())
        .cloned()
        .collect::<Vec<_>>()
        .join(" || ")
}

fn default_if_empty(text: &str, fallback: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        fallback.to_string()
    } else {
        trimmed.to_string()
    }
}

fn make_builder_prompt(
    question: &str,
    glossary: &str,
    delta_g: &str,
    reference_b_msg: Option<&str>,
    reference_b_codes: &str,
    last_c_sum: &str,
    last_c_decode: &str,
    last_c_audit: &str,
    is_opening: bool,
) -> String {
    let last_b = reference_b_msg
        .map(|text| shorten_for_prompt(text, MAX_CONTEXT_SNIPPET))
        .unwrap_or_else(|| "(no challenger input yet)".to_string());
    let b_codes = if reference_b_codes.is_empty() {
        "(none)".to_string()
    } else {
        reference_b_codes.to_string()
    };
    let guidance = if is_opening {
        "Open the round with fresh builder insights. Provide at least one new angle beyond prior rounds and set direction.".to_string()
    } else {
        "Answer B directly. Address at least two of B's codes, add one fresh constructive move, and keep momentum.".to_string()
    };

    format!(
        "TOPIC: {question}\nGLOSSARY: {glossary}\nDELTA_G_CURRENT: {delta_g}\nC_SUMMARY_LAST: {sum}\nC_DECODE_LAST: {decode}\nC_AUDIT_LAST: {audit}\nLAST_B_MESSAGE: {last_b}\nLAST_B_CODES: {b_codes}\nROLE_GOAL: {guidance}\nOUTPUT_RULES: Ultra-concise short-code stream (<=35 tokens, >=70% glossary codes). Max 2 new code proposals via 'DELTA+ CODE:desc'. Finish with decisive next action.",
        question = question,
        glossary = glossary,
        delta_g = default_if_empty(delta_g, "(none)"),
        sum = default_if_empty(last_c_sum, "(none yet)"),
        decode = default_if_empty(last_c_decode, "(none yet)"),
        audit = default_if_empty(last_c_audit, "(none yet)"),
        last_b = last_b,
        b_codes = b_codes,
        guidance = guidance
    )
}

fn make_challenger_prompt(
    question: &str,
    glossary: &str,
    delta_g: &str,
    reference_a_msg: Option<&str>,
    reference_a_codes: &str,
    last_c_sum: &str,
    last_c_decode: &str,
    last_c_audit: &str,
    is_opening: bool,
) -> String {
    let last_a = reference_a_msg
        .map(|text| shorten_for_prompt(text, MAX_CONTEXT_SNIPPET))
        .unwrap_or_else(|| "(no builder input yet)".to_string());
    let a_codes = if reference_a_codes.is_empty() {
        "(none)".to_string()
    } else {
        reference_a_codes.to_string()
    };
    let guidance = if is_opening {
        "Open the round with a critical angle. Surface risks or blind spots without repeating older material.".to_string()
    } else {
        "Directly counter at least two of A's codes. Add one or two risk-driven insights that stress-test the plan.".to_string()
    };

    format!(
        "TOPIC: {question}\nGLOSSARY: {glossary}\nDELTA_G_CURRENT: {delta_g}\nC_SUMMARY_LAST: {sum}\nC_DECODE_LAST: {decode}\nC_AUDIT_LAST: {audit}\nLAST_A_MESSAGE: {last_a}\nLAST_A_CODES: {a_codes}\nROLE_GOAL: {guidance}\nOUTPUT_RULES: Ultra-concise short-code stream (<=35 tokens, >=70% glossary codes). Max 2 new code proposals via 'DELTA+ CODE:desc'. Make tension and tests explicit.",
        question = question,
        glossary = glossary,
        delta_g = default_if_empty(delta_g, "(none)"),
        sum = default_if_empty(last_c_sum, "(none yet)"),
        decode = default_if_empty(last_c_decode, "(none yet)"),
        audit = default_if_empty(last_c_audit, "(none yet)"),
        last_a = last_a,
        a_codes = a_codes,
        guidance = guidance
    )
}

fn make_c_start_prompt(question: &str) -> String {
    format!(
        "TOPIC: {question}. Create only guardrails and a small starter glossary (2-4 neutral codes). Do not produce a final answer. Reply strictly in JSON with keys DELTA_G, AUDIT, SUM, DECODE, GLOSSARY.",
        question = question
    )
}

fn make_c_round_prompt(
    question: &str,
    a_msg: &str,
    b_msg: &str,
    glossary: &str,
    delta_g: &str,
    delta_g_history: &[String],
) -> String {
    format!(
        "TOPIC: {question}\nMESSAGE_A: {a}\nMESSAGE_B: {b}\nGLOSSARY: {glossary}\nDELTA_G_PREVIOUS: {delta}\nDELTA_G_HISTORY_RECENT: {history}\nTASK: Evaluate code proposals 'DELTA+ CODE:desc'. Accept up to two, reject invalid ones with '-CODE'. Update the glossary text. Reply with JSON using keys DELTA_G, AUDIT, SUM, DECODE, GLOSSARY only.",
        question = question,
        a = shorten_for_prompt(a_msg, MAX_CONTEXT_SNIPPET),
        b = shorten_for_prompt(b_msg, MAX_CONTEXT_SNIPPET),
        glossary = glossary,
        delta = default_if_empty(delta_g, "(none)"),
        history = default_if_empty(&recent_history_snippet(delta_g_history, 20), "(none)"),
    )
}

fn make_c_final_prompt(question: &str, glossary: &str, final_a: &str, final_b: &str) -> String {
    format!(
        "TOPIC: {question}\nGLOSSARY: {glossary}\nFINAL_A: {a}\nFINAL_B: {b}\nTASK: Reply with JSON (DELTA_G, AUDIT, SUM, DECODE, GLOSSARY, FINAL_ANSWER). FINAL_ANSWER must be 3-5 neutral German sentences synthesising the debate.",
        question = question,
        glossary = glossary,
        a = shorten_for_prompt(final_a, MAX_CONTEXT_SNIPPET),
        b = shorten_for_prompt(final_b, MAX_CONTEXT_SNIPPET),
    )
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    let api_key = std::env::var("DEEPSEEK_API_KEY")
        .context("Please set the DEEPSEEK_API_KEY environment variable")?;

    print!("Enter your topic/question: ");
    io::stdout().flush().ok();
    let mut question = String::new();
    io::stdin()
        .read_line(&mut question)
        .context("Failed to read input")?;
    let question = question.trim().to_string();

    let client = Client::new();

    println!("\nTopic: {question}", question = question);
    println!(
        "Starting 3-agent debate ({rounds} rounds): A=Builder, B=Challenger, C=Librarian\n",
        rounds = DEBATE_ROUNDS
    );

    let (
        mut glossary,
        mut delta_g,
        mut delta_g_history,
        mut last_c_sum,
        mut last_c_decode,
        mut last_c_audit,
    ) = if let Some(state) = load_state() {
        println!("Loaded state from {}", STATE_FILE);
        (
            state.glossary,
            state.delta_g,
            state.delta_g_history,
            String::new(),
            String::new(),
            String::new(),
        )
    } else {
        let c_start_json = complete_chat_with_tokens(
            &client,
            &api_key,
            vec![
                librarian_system_prompt(),
                ChatMessage {
                    role: "user".to_string(),
                    content: make_c_start_prompt(&question),
                },
            ],
            Some(MAX_TOKENS_C_START),
            Some(TEMP_C),
        )
        .await?;

        let c_start = parse_librarian_response(&c_start_json)
            .context("Failed to parse librarian start response")?;

        println!("C (start) DELTA_G: {}", c_start.delta_g);
        println!("C (start) AUDIT: {}\n", c_start.audit);

        (
            c_start.glossary.clone(),
            c_start.delta_g.clone(),
            vec![c_start.delta_g.clone()],
            c_start.sum,
            c_start.decode,
            c_start.audit,
        )
    };

    let mut last_a_msg = String::new();
    let mut last_b_msg = String::new();

    for round in 1..=DEBATE_ROUNDS {
        println!("--- Round {} ---", round);

        let a_goes_first = round % 2 == 1;
        let (a_msg, b_msg) = if a_goes_first {
            let builder_prompt = make_builder_prompt(
                &question,
                &glossary,
                &delta_g,
                if last_b_msg.is_empty() {
                    None
                } else {
                    Some(&last_b_msg)
                },
                &extract_codes(&last_b_msg).join(", "),
                &last_c_sum,
                &last_c_decode,
                &last_c_audit,
                true,
            );
            print!("A (round {round}): ", round = round);
            io::stdout().flush().ok();
            let a_msg_cur = stream_chat_with_tokens(
                &client,
                &api_key,
                vec![
                    builder_system_prompt(),
                    ChatMessage {
                        role: "user".to_string(),
                        content: builder_prompt,
                    },
                ],
                Some(MAX_TOKENS_AB_MSG),
                Some(TEMP_AB),
            )
            .await?;

            let challenger_prompt = make_challenger_prompt(
                &question,
                &glossary,
                &delta_g,
                Some(&a_msg_cur),
                &extract_codes(&a_msg_cur).join(", "),
                &last_c_sum,
                &last_c_decode,
                &last_c_audit,
                false,
            );
            print!("\nB (round {round}): ", round = round);
            io::stdout().flush().ok();
            let b_msg_cur = stream_chat_with_tokens(
                &client,
                &api_key,
                vec![
                    challenger_system_prompt(),
                    ChatMessage {
                        role: "user".to_string(),
                        content: challenger_prompt,
                    },
                ],
                Some(MAX_TOKENS_AB_MSG),
                Some(TEMP_AB),
            )
            .await?;
            println!("");
            (a_msg_cur, b_msg_cur)
        } else {
            let challenger_prompt = make_challenger_prompt(
                &question,
                &glossary,
                &delta_g,
                if last_a_msg.is_empty() {
                    None
                } else {
                    Some(&last_a_msg)
                },
                &extract_codes(&last_a_msg).join(", "),
                &last_c_sum,
                &last_c_decode,
                &last_c_audit,
                true,
            );
            print!("B (round {round}): ", round = round);
            io::stdout().flush().ok();
            let b_msg_cur = stream_chat_with_tokens(
                &client,
                &api_key,
                vec![
                    challenger_system_prompt(),
                    ChatMessage {
                        role: "user".to_string(),
                        content: challenger_prompt,
                    },
                ],
                Some(MAX_TOKENS_AB_MSG),
                Some(TEMP_AB),
            )
            .await?;

            let builder_prompt = make_builder_prompt(
                &question,
                &glossary,
                &delta_g,
                Some(&b_msg_cur),
                &extract_codes(&b_msg_cur).join(", "),
                &last_c_sum,
                &last_c_decode,
                &last_c_audit,
                false,
            );
            print!("\nA (round {round}): ", round = round);
            io::stdout().flush().ok();
            let a_msg_cur = stream_chat_with_tokens(
                &client,
                &api_key,
                vec![
                    builder_system_prompt(),
                    ChatMessage {
                        role: "user".to_string(),
                        content: builder_prompt,
                    },
                ],
                Some(MAX_TOKENS_AB_MSG),
                Some(TEMP_AB),
            )
            .await?;
            println!("");
            (a_msg_cur, b_msg_cur)
        };

        let c_round_json = complete_chat_with_tokens(
            &client,
            &api_key,
            vec![
                librarian_system_prompt(),
                ChatMessage {
                    role: "user".to_string(),
                    content: make_c_round_prompt(
                        &question,
                        &a_msg,
                        &b_msg,
                        &glossary,
                        &delta_g,
                        &delta_g_history,
                    ),
                },
            ],
            Some(MAX_TOKENS_C_ROUND),
            Some(TEMP_C),
        )
        .await?;

        let c_round = match parse_librarian_response(&c_round_json) {
            Ok(response) => response,
            Err(err) => {
                eprintln!("Warning: could not parse librarian round JSON: {}", err);
                LibrarianResponse::default()
            }
        };

        println!(
            "C (round {round}) AUDIT: {audit}",
            round = round,
            audit = c_round.audit
        );
        println!(
            "C (round {round}) DELTA_G: {delta}",
            round = round,
            delta = c_round.delta_g
        );
        println!("");

        let reconciled_glossary = reconcile_glossary(&glossary, &c_round.delta_g);
        if reconciled_glossary != glossary {
            println!("Glossary updated: {}", reconciled_glossary);
        }

        glossary = reconciled_glossary;
        delta_g = c_round.delta_g.clone();
        last_c_sum = c_round.sum;
        last_c_decode = c_round.decode;
        last_c_audit = c_round.audit;

        if !delta_g.trim().is_empty() {
            if delta_g_history
                .last()
                .map(|entry| entry == &delta_g)
                .unwrap_or(false)
                == false
            {
                delta_g_history.push(delta_g.clone());
            }
            if delta_g_history.len() > 200 {
                let start = delta_g_history.len() - 200;
                delta_g_history = delta_g_history[start..].to_vec();
            }
        }

        last_a_msg = a_msg;
        last_b_msg = b_msg;
    }

    let c_final_json = complete_chat_with_tokens(
        &client,
        &api_key,
        vec![
            librarian_system_prompt(),
            ChatMessage {
                role: "user".to_string(),
                content: make_c_final_prompt(&question, &glossary, &last_a_msg, &last_b_msg),
            },
        ],
        Some(MAX_TOKENS_C_FINAL),
        Some(TEMP_C),
    )
    .await?;

    let c_final = match parse_librarian_response(&c_final_json) {
        Ok(response) => response,
        Err(err) => {
            eprintln!("Warning: could not parse final librarian JSON: {}", err);
            LibrarianResponse {
                delta_g: String::new(),
                audit: String::from("parse-error"),
                sum: String::new(),
                decode: String::new(),
                glossary: glossary.clone(),
                final_answer: Some("Final summary not available (invalid JSON).".to_string()),
            }
        }
    };

    println!("\n===== Final Librarian Output =====");
    println!("DELTA_G: {}", c_final.delta_g);
    println!("AUDIT: {}", c_final.audit);
    println!("SUM: {}", c_final.sum);
    println!("DECODE: {}", c_final.decode);
    println!("GLOSSARY: {}", c_final.glossary);

    if let Some(answer) = c_final.final_answer {
        println!("\nFinal answer:\n{}", answer);
    } else {
        println!("\nNote: FINAL_ANSWER missing in librarian reply.");
    }

    let state = PersistentState {
        glossary,
        delta_g,
        delta_g_history,
    };
    if let Err(err) = save_state(&state) {
        eprintln!("Warning: could not persist state: {}", err);
    } else {
        println!("\nState stored in {}", STATE_FILE);
    }

    Ok(())
}
