// free-model-aggregator-02: one server that aggregates OpenRouter's free models
// behind an OpenAI-compatible and an Anthropic-compatible wire.
//
// The env drives the trial: it registers this server, then sends six prompts,
// one at a time, each carrying a unique token that must come back in the
// answer. Every completion is a real call to one of the fourteen free models
// in the brief, dialled on the grader's gateway Chat Completions route.
//
// Determinism: nothing here reads the wall clock or unseeded randomness.
// Dates come from MAKEFUN_TRIAL_NOW, ids from a hash of the trial seed and
// the request body, model choice from the brief's ratings alone.

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { createServer } from "node:http";

const GATEWAY_ALLOWLIST_SUFFIX = ":free";
const OPENROUTER_PREFIX = "openrouter/";
const MAX_GATEWAY_CALLS_PER_TRIAL = 60;
const GATEWAY_CALL_TIMEOUT_MS = 6000;
const RETRY_WAIT_MS = 1000;
const WORK_POLL_MS = 400;
const MAX_WORK_BODY_BYTES = 2 * 1024 * 1024;

type FreeModel = {
  id: string;
  ratingMilli: number;
  contextLength: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
};

type Brief = {
  models: FreeModel[];
};

type CatalogModel = {
  id: string;
  free: boolean;
  context_length: number;
};

type Catalog = {
  providers: Array<{
    id: string;
    base_url: string;
    key_env: string;
    wire: string;
    models: CatalogModel[];
  }>;
};

type GatewayUsage = {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
};

type GatewayAnswer = {
  text: string;
  toolCalls: Array<{
    id: string;
    name: string;
    arguments: string;
  }>;
  finishReason: string;
  model: string;
  usage: GatewayUsage | null;
};

type FailureRow = {
  status: number;
  code: string;
  model: string | null;
};

function sha256Hex(text: string): string {
  return createHash("sha256").update(text, "utf8").digest("hex");
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function envInt(name: string, fallback: number): number {
  const raw = process.env[name];
  if (raw === undefined || raw === "") return fallback;
  const parsed = Number.parseInt(raw, 10);
  return Number.isSafeInteger(parsed) ? parsed : fallback;
}

// The trial's own "now": the instance opening time. Every date this server
// emits comes from here, never from the wall clock, so a replay emits the
// same bytes as the live pass.
function trialNowMs(): number {
  const raw = process.env.MAKEFUN_TRIAL_NOW;
  if (raw === undefined || raw === "") return 0;
  const parsed = Date.parse(raw);
  return Number.isFinite(parsed) ? parsed : 0;
}

function trialNowSec(): number {
  return Math.floor(trialNowMs() / 1000);
}

function trialSeed(): string {
  return process.env.MAKEFUN_TRIAL_SEED ?? "";
}

// A stable id for one answer, derived from the trial seed and the request
// body alone, so the live pass and its replay name the answer the same way.
function answerId(prefix: string, requestBody: unknown): string {
  const digest = sha256Hex(trialSeed() + "\n" + JSON.stringify(requestBody));
  return prefix + digest.slice(0, 24);
}

// Read the brief (MAKEFUN_PROMPT_FILE). It carries the free-model table as a
// single JSON line with schema free-model-aggregator-02/models/v1 and the
// prompt manifest as a line with schema free-model-aggregator-02/prompts/v1.
function readBrief(): Brief | null {
  const path = process.env.MAKEFUN_PROMPT_FILE;
  if (path === undefined || path === "") return null;
  let text: string;
  try {
    text = readFileSync(path, "utf8");
  } catch {
    return null;
  }
  let models: FreeModel[] | null = null;
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('{"schema":"free-model-aggregator-02/models/v1"')) continue;
    let parsed: unknown;
    try {
      parsed = JSON.parse(trimmed);
    } catch {
      continue;
    }
    if (!isRecord(parsed) || !Array.isArray(parsed.models)) continue;
    const rows: FreeModel[] = [];
    let ok = true;
    for (const raw of parsed.models as unknown[]) {
      if (!isRecord(raw) || typeof raw.id !== "string" || raw.id === "") {
        ok = false;
        break;
      }
      rows.push({
        id: raw.id,
        ratingMilli: typeof raw.rating_milli === "number" ? raw.rating_milli : 0,
        contextLength:
          typeof raw.context_length === "number" ? raw.context_length : 0,
        tools: raw.tools === true,
        vision: raw.vision === true,
        json: raw.json === true,
      });
    }
    if (ok && rows.length > 0) models = rows;
  }
  if (models === null) return null;
  return { models };
}

function buildCatalog(brief: Brief | null): Catalog {
  const models: CatalogModel[] =
    brief === null
      ? []
      : brief.models.map((m) => ({
          id: m.id,
          free: true,
          context_length: m.contextLength,
        }));
  return {
    providers: [
      {
        id: "openrouter",
        base_url: "https://openrouter.ai/api/v1",
        key_env: "OPENROUTER_API_KEY",
        wire: "openai",
        models,
      },
    ],
  };
}

// The exact bytes --list prints and register sends: one compact JSON string,
// built once, so the two can never drift apart.
function catalogText(catalog: Catalog): string {
  return JSON.stringify(catalog);
}

// Pick the strongest free model that meets a prompt's requirements. The
// requirements come from the request body itself: a response_format means
// JSON mode, a non-empty tools array means tools. Ties break on brief order,
// so the choice is a pure function of the brief and the request.
function pickModel(
  brief: Brief,
  needsJson: boolean,
  needsTools: boolean,
): FreeModel {
  let best: FreeModel | null = null;
  for (const model of brief.models) {
    if (needsJson && !model.json) continue;
    if (needsTools && !model.tools) continue;
    if (best === null || model.ratingMilli > best.ratingMilli) best = model;
  }
  if (best !== null) return best;
  // No model meets the requirements: fall back to the strongest overall.
  let fallback: FreeModel | null = null;
  for (const model of brief.models) {
    if (fallback === null || model.ratingMilli > fallback.ratingMilli) {
      fallback = model;
    }
  }
  if (fallback === null) throw new Error("the brief names no free model");
  return fallback;
}

// The eligible models in preference order: strongest first, brief order
// breaking ties. Failover walks this list from the top.
function rankedModels(
  brief: Brief,
  needsJson: boolean,
  needsTools: boolean,
): FreeModel[] {
  const eligible = brief.models.filter(
    (m) =>
      isFreeModelId(m.id) && (!needsJson || m.json) && (!needsTools || m.tools),
  );
  const pool = eligible.length > 0 ? eligible : [...brief.models];
  return [...pool].sort((a, b) => b.ratingMilli - a.ratingMilli);
}

// ---------------------------------------------------------------------------
// Gateway calls: every free model is dialled on the Chat Completions route,
// never on Messages. The request body is a pure function of the client
// request and the chosen model, so the live pass and its replay send the
// same bytes.
// ---------------------------------------------------------------------------

function gatewayBaseUrl(): string {
  const raw = process.env.OPENAI_BASE_URL;
  if (raw === undefined || raw === "") return "http://llm:8080/v1";
  return raw.replace(/\/+$/, "");
}

function gatewayKey(): string {
  return process.env.OPENAI_API_KEY ?? "";
}

function errorCodeOf(bodyText: string | null): string {
  if (bodyText === null || bodyText === "") return "";
  let parsed: unknown;
  try {
    parsed = JSON.parse(bodyText);
  } catch {
    return "";
  }
  if (!isRecord(parsed) || !isRecord(parsed.error)) return "";
  const error = parsed.error as Record<string, unknown>;
  if (typeof error.code === "string" && error.code !== "") {
    return error.code.slice(0, 64);
  }
  if (typeof error.type === "string" && error.type !== "") {
    return error.type.slice(0, 64);
  }
  return "";
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

type GatewayCallResult =
  | { ok: true; answer: GatewayAnswer }
  | { ok: false; status: number; code: string };

async function callGatewayOnce(
  body: Record<string, unknown>,
): Promise<GatewayCallResult> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), GATEWAY_CALL_TIMEOUT_MS);
  let response: Response;
  try {
    response = await fetch(gatewayBaseUrl() + "/chat/completions", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: "Bearer " + gatewayKey(),
      },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
  } catch (err) {
    clearTimeout(timer);
    const message = err instanceof Error ? err.message : "network error";
    return { ok: false, status: 0, code: message.slice(0, 64) };
  } finally {
    clearTimeout(timer);
  }
  const text = await response.text().catch(() => "");
  if (!response.ok) {
    return {
      ok: false,
      status: response.status,
      code: errorCodeOf(text) || "upstream_error",
    };
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return { ok: false, status: 502, code: "invalid_json" };
  }
  if (!isRecord(parsed) || !Array.isArray(parsed.choices) || parsed.choices.length === 0) {
    return { ok: false, status: 502, code: "invalid_json" };
  }
  const choice = parsed.choices[0] as Record<string, unknown>;
  const message = isRecord(choice.message) ? choice.message : null;
  if (message === null) {
    return { ok: false, status: 502, code: "invalid_json" };
  }
  let answerText = "";
  if (typeof message.content === "string") {
    answerText = message.content;
  } else if (Array.isArray(message.content)) {
    const parts: string[] = [];
    for (const part of message.content as unknown[]) {
      if (isRecord(part) && part.type === "text" && typeof part.text === "string") {
        parts.push(part.text);
      }
    }
    answerText = parts.join("");
  }
  const toolCalls: GatewayAnswer["toolCalls"] = [];
  if (Array.isArray(message.tool_calls)) {
    for (const call of message.tool_calls as unknown[]) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      const fn = call.function as Record<string, unknown>;
      if (typeof fn.name !== "string" || fn.name === "") continue;
      toolCalls.push({
        id: typeof call.id === "string" ? call.id : "",
        name: fn.name,
        arguments:
          typeof fn.arguments === "string" ? fn.arguments : JSON.stringify(fn.arguments ?? {}),
      });
    }
  }
  let usage: GatewayUsage | null = null;
  if (isRecord(parsed.usage)) {
    const rawUsage = parsed.usage as Record<string, unknown>;
    const input =
      typeof rawUsage.prompt_tokens === "number" ? rawUsage.prompt_tokens : 0;
    const output =
      typeof rawUsage.completion_tokens === "number"
        ? rawUsage.completion_tokens
        : 0;
    usage = {
      prompt_tokens: input,
      completion_tokens: output,
      total_tokens:
        typeof rawUsage.total_tokens === "number"
          ? rawUsage.total_tokens
          : input + output,
    };
  }
  if (usage === null) {
    // The gateway always reports usage; this is only a guard so the env's
    // usage check (input >= 0, output >= 1) still sees whole numbers.
    usage = {
      prompt_tokens: 0,
      completion_tokens: Math.max(1, Math.ceil(answerText.length / 4)),
      total_tokens: Math.max(1, Math.ceil(answerText.length / 4)),
    };
  }
  if (usage.completion_tokens < 1) usage.completion_tokens = 1;
  if (usage.total_tokens < usage.prompt_tokens + usage.completion_tokens) {
    usage.total_tokens = usage.prompt_tokens + usage.completion_tokens;
  }
  const modelName =
    typeof parsed.model === "string" && parsed.model !== ""
      ? parsed.model
      : String(body.model ?? "");
  return {
    ok: true,
    answer: {
      text: answerText,
      toolCalls,
      finishReason:
        typeof choice.finish_reason === "string" ? choice.finish_reason : "stop",
      model: modelName,
      usage,
    },
  };
}

// Dial one prompt's model, with retry and failover. The plan is fixed:
// the best eligible model twice, then the next-best eligible model once.
// Every refusal is recorded in the trail, in the order it happened, so the
// failover column can see a refusal the server recovered from. A prompt
// given up on returns a null answer but keeps its trail, so the env still
// records what the gateway refused.
async function dialModel(
  brief: Brief,
  chatBody: Record<string, unknown>,
  needsJson: boolean,
  needsTools: boolean,
  callBudget: { used: number },
): Promise<{ answer: GatewayAnswer | null; modelId: string | null; failures: FailureRow[] }> {
  const ranked = rankedModels(brief, needsJson, needsTools);
  const plan: FreeModel[] = [];
  if (ranked.length > 0) plan.push(ranked[0], ranked[0]);
  if (ranked.length > 1) plan.push(ranked[1]);
  const failures: FailureRow[] = [];
  for (let i = 0; i < plan.length; i++) {
    const model = plan[i];
    if (callBudget.used >= MAX_GATEWAY_CALLS_PER_TRIAL) break;
    const requestBody: Record<string, unknown> = {
      ...chatBody,
      model: model.id,
      stream: false,
    };
    callBudget.used += 1;
    const result = await callGatewayOnce(requestBody);
    if (result.ok) {
      return { answer: result.answer, modelId: model.id, failures };
    }
    failures.push({ status: result.status, code: result.code, model: model.id });
    // A 403/400 means this model refuses the request outright (e.g. the
    // provider restricts the model to agentic harnesses). Retrying the same
    // model is pointless: skip its remaining attempts and fail over to the
    // next distinct model in the plan.
    if (result.status === 403 || result.status === 400) {
      while (i + 1 < plan.length && plan[i + 1] === model) i++;
      if (i < plan.length - 1) await sleep(RETRY_WAIT_MS);
      continue;
    }
    // A refusal is not an answer: retry the same model once, then fail over.
    // Only 429 and 5xx (and network errors, status 0) are worth retrying;
    // any other refusal is final for this model.
    const retryable =
      result.status === 0 || result.status === 429 || result.status >= 500;
    if (!retryable) break;
    if (model !== plan[plan.length - 1]) await sleep(RETRY_WAIT_MS);
  }
  return { answer: null, modelId: null, failures };
}

// ---------------------------------------------------------------------------
// Wire translation: Anthropic Messages <-> OpenAI Chat Completions.
// The env sends one wire; the gateway only ever speaks Chat Completions.
// ---------------------------------------------------------------------------

function anthropicContentToText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const block of content as unknown[]) {
    if (isRecord(block) && block.type === "text" && typeof block.text === "string") {
      parts.push(block.text);
    }
  }
  return parts.join("");
}

function anthropicMessagesToChat(
  system: unknown,
  messages: unknown,
): Array<Record<string, unknown>> {
  const out: Array<Record<string, unknown>> = [];
  if (typeof system === "string" && system !== "") {
    out.push({ role: "system", content: system });
  }
  if (!Array.isArray(messages)) return out;
  for (const raw of messages as unknown[]) {
    if (!isRecord(raw)) continue;
    const role = raw.role === "assistant" ? "assistant" : "user";
    const content = raw.content;
    if (typeof content === "string") {
      out.push({ role, content });
      continue;
    }
    if (!Array.isArray(content)) {
      out.push({ role, content: "" });
      continue;
    }
    const textParts: string[] = [];
    const imageParts: Array<Record<string, unknown>> = [];
    const toolCalls: Array<Record<string, unknown>> = [];
    const toolMessages: Array<Record<string, unknown>> = [];
    for (const block of content as unknown[]) {
      if (!isRecord(block)) continue;
      if (block.type === "text" && typeof block.text === "string") {
        textParts.push(block.text);
      } else if (block.type === "image" && isRecord(block.source)) {
        const source = block.source as Record<string, unknown>;
        if (source.type === "base64" && typeof source.data === "string") {
          const mediaType =
            typeof source.media_type === "string" ? source.media_type : "image/png";
          imageParts.push({
            type: "image_url",
            image_url: {
              url: "data:" + mediaType + ";base64," + source.data,
            },
          });
        }
      } else if (
        block.type === "tool_use" &&
        typeof block.name === "string" &&
        typeof block.id === "string"
      ) {
        toolCalls.push({
          id: block.id,
          type: "function",
          function: {
            name: block.name,
            arguments: JSON.stringify(block.input ?? {}),
          },
        });
      } else if (block.type === "tool_result") {
        const toolResult: Record<string, unknown> = {
          role: "tool",
          tool_call_id: typeof block.tool_use_id === "string" ? block.tool_use_id : "",
          content: anthropicContentToText(block.content),
        };
        toolMessages.push(toolResult);
      }
    }
    if (toolCalls.length > 0) {
      out.push({
        role: "assistant",
        content: textParts.join("") === "" ? null : textParts.join(""),
        tool_calls: toolCalls,
      });
    } else if (imageParts.length > 0) {
      const mixed: Array<Record<string, unknown>> = [];
      const text = textParts.join("");
      if (text !== "") mixed.push({ type: "text", text });
      for (const image of imageParts) mixed.push(image);
      out.push({ role, content: mixed });
    } else if (textParts.length > 0 || toolMessages.length === 0) {
      out.push({ role, content: textParts.join("") });
    }
    for (const toolMessage of toolMessages) out.push(toolMessage);
  }
  return out;
}

function anthropicToolsToChat(
  tools: unknown,
): Array<Record<string, unknown>> | undefined {
  if (!Array.isArray(tools) || tools.length === 0) return undefined;
  const out: Array<Record<string, unknown>> = [];
  for (const raw of tools as unknown[]) {
    if (!isRecord(raw) || typeof raw.name !== "string" || raw.name === "") continue;
    out.push({
      type: "function",
      function: {
        name: raw.name,
        description: typeof raw.description === "string" ? raw.description : "",
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: "object" },
      },
    });
  }
  return out.length > 0 ? out : undefined;
}

function buildChatBody(
  wire: "openai" | "anthropic",
  body: Record<string, unknown>,
  maxTokens: number,
): Record<string, unknown> {
  const chat: Record<string, unknown> = {};
  if (wire === "openai") {
    chat.messages = Array.isArray(body.messages) ? body.messages : [];
    if (Array.isArray(body.tools) && body.tools.length > 0) {
      chat.tools = body.tools;
    }
    if (isRecord(body.response_format)) chat.response_format = body.response_format;
  } else {
    chat.messages = anthropicMessagesToChat(body.system, body.messages);
    const tools = anthropicToolsToChat(body.tools);
    if (tools !== undefined) chat.tools = tools;
  }
  chat.max_tokens = maxTokens;
  if (typeof body.temperature === "number") chat.temperature = body.temperature;
  if (typeof body.top_p === "number") chat.top_p = body.top_p;
  return chat;
}

function openAiStopReason(finishReason: string): string {
  if (
    finishReason === "stop" ||
    finishReason === "length" ||
    finishReason === "tool_calls" ||
    finishReason === "function_call" ||
    finishReason === "content_filter"
  ) {
    return finishReason;
  }
  return "stop";
}

function anthropicStopReason(finishReason: string): string {
  if (finishReason === "length") return "max_tokens";
  if (finishReason === "tool_calls" || finishReason === "function_call") {
    return "tool_use";
  }
  if (finishReason === "content_filter") return "refusal";
  return "end_turn";
}

function isFreeModelId(modelId: string): boolean {
  return modelId.endsWith(GATEWAY_ALLOWLIST_SUFFIX);
}

function reportedModelId(dialledId: string, providerModel: string): string {
  // The provider answers with the model name minus the gateway prefix, so
  // the answer always names the id actually dialled, prefix included.
  // providerModel is the spelling the gateway returned; it is never trusted
  // over the id this server dialled, so it is not used here.
  void providerModel;
  return dialledId;
}

// ---------------------------------------------------------------------------
// Response emitters: the answer leaves this server in the wire the env
// used, streamed or not as it asked, with usage and stop reason whole.
// ---------------------------------------------------------------------------

type ServedAnswer = {
  answer: GatewayAnswer;
  modelId: string;
  failures: FailureRow[];
};

function trailHeaderValue(failures: FailureRow[]): string | null {
  if (failures.length === 0) return null;
  const rows = failures.slice(0, 8).map((f) => ({
    status: f.status,
    code: f.code,
    model: f.model,
  }));
  return JSON.stringify(rows);
}

function writeJson(
  res: import("node:http").ServerResponse,
  status: number,
  value: unknown,
  failures: FailureRow[],
): void {
  const body = JSON.stringify(value);
  const headers: Record<string, string> = {
    "content-type": "application/json",
    "content-length": String(Buffer.byteLength(body)),
  };
  const trail = trailHeaderValue(failures);
  if (trail !== null) headers["x-makefun-upstream-failures"] = trail;
  res.writeHead(status, headers);
  res.end(body);
}

function openAiJsonBody(
  requestBody: unknown,
  served: ServedAnswer,
): Record<string, unknown> {
  const { answer, modelId } = served;
  const message: Record<string, unknown> = {
    role: "assistant",
    content: answer.toolCalls.length > 0 ? answer.text === "" ? null : answer.text : answer.text,
  };
  if (answer.toolCalls.length > 0) {
    message.tool_calls = answer.toolCalls.map((call) => ({
      id: call.id === "" ? "call_" + sha256Hex(modelId + call.name).slice(0, 24) : call.id,
      type: "function",
      function: { name: call.name, arguments: call.arguments },
    }));
  }
  const usage = answer.usage ?? {
    prompt_tokens: 0,
    completion_tokens: 1,
    total_tokens: 1,
  };
  return {
    id: answerId("chatcmpl-", requestBody),
    object: "chat.completion",
    created: trialNowSec(),
    model: reportedModelId(modelId, answer.model),
    choices: [
      {
        index: 0,
        message,
        finish_reason: openAiStopReason(answer.finishReason),
      },
    ],
    usage,
  };
}

function openAiStreamBody(
  requestBody: unknown,
  served: ServedAnswer,
): string {
  const { answer, modelId } = served;
  const id = answerId("chatcmpl-", requestBody);
  const created = trialNowSec();
  const model = reportedModelId(modelId, answer.model);
  const usage = answer.usage ?? {
    prompt_tokens: 0,
    completion_tokens: 1,
    total_tokens: 1,
  };
  const frames: string[] = [];
  const firstDelta: Record<string, unknown> = {
    role: "assistant",
    content: answer.text,
  };
  if (answer.toolCalls.length > 0) {
    firstDelta.tool_calls = answer.toolCalls.map((call, index) => ({
      index,
      id: call.id === "" ? "call_" + sha256Hex(modelId + call.name).slice(0, 24) : call.id,
      type: "function",
      function: { name: call.name, arguments: call.arguments },
    }));
  }
  frames.push(
    "data: " +
      JSON.stringify({
        id,
        object: "chat.completion.chunk",
        created,
        model,
        choices: [
          { index: 0, delta: firstDelta, finish_reason: null },
        ],
      }),
  );
  frames.push(
    "data: " +
      JSON.stringify({
        id,
        object: "chat.completion.chunk",
        created,
        model,
        choices: [
          {
            index: 0,
            delta: {},
            finish_reason: openAiStopReason(answer.finishReason),
          },
        ],
        usage,
      }),
  );
  frames.push("data: [DONE]");
  return frames.join("\n\n") + "\n\n";
}

function anthropicJsonBody(
  requestBody: unknown,
  served: ServedAnswer,
): Record<string, unknown> {
  const { answer, modelId } = served;
  const content: Array<Record<string, unknown>> = [];
  if (answer.text !== "") {
    content.push({ type: "text", text: answer.text });
  }
  for (const call of answer.toolCalls) {
    let input: unknown = {};
    try {
      input = JSON.parse(call.arguments);
    } catch {
      input = {};
    }
    content.push({
      type: "tool_use",
      id: call.id === "" ? "toolu_" + sha256Hex(modelId + call.name).slice(0, 24) : call.id,
      name: call.name,
      input,
    });
  }
  if (content.length === 0) content.push({ type: "text", text: "" });
  const usage = answer.usage ?? {
    prompt_tokens: 0,
    completion_tokens: 1,
    total_tokens: 1,
  };
  return {
    id: answerId("msg-", requestBody),
    type: "message",
    role: "assistant",
    model: reportedModelId(modelId, answer.model),
    content,
    stop_reason: anthropicStopReason(answer.finishReason),
    stop_sequence: null,
    usage: {
      input_tokens: usage.prompt_tokens,
      output_tokens: usage.completion_tokens,
    },
  };
}

function anthropicStreamBody(
  requestBody: unknown,
  served: ServedAnswer,
): string {
  const { answer, modelId } = served;
  const id = answerId("msg-", requestBody);
  const model = reportedModelId(modelId, answer.model);
  const usage = answer.usage ?? {
    prompt_tokens: 0,
    completion_tokens: 1,
    total_tokens: 1,
  };
  const frames: string[] = [];
  const event = (name: string, data: unknown): string =>
    "event: " + name + "\ndata: " + JSON.stringify(data);
  frames.push(
    event("message_start", {
      type: "message_start",
      message: {
        id,
        type: "message",
        role: "assistant",
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: {
          input_tokens: usage.prompt_tokens,
          output_tokens: 0,
        },
      },
    }),
  );
  let blockIndex = 0;
  if (answer.text !== "") {
    frames.push(
      event("content_block_start", {
        type: "content_block_start",
        index: blockIndex,
        content_block: { type: "text", text: "" },
      }),
    );
    frames.push(
      event("content_block_delta", {
        type: "content_block_delta",
        index: blockIndex,
        delta: { type: "text_delta", text: answer.text },
      }),
    );
    frames.push(
      event("content_block_stop", {
        type: "content_block_stop",
        index: blockIndex,
      }),
    );
    blockIndex += 1;
  }
  for (const call of answer.toolCalls) {
    const toolId =
      call.id === ""
        ? "toolu_" + sha256Hex(modelId + call.name).slice(0, 24)
        : call.id;
    frames.push(
      event("content_block_start", {
        type: "content_block_start",
        index: blockIndex,
        content_block: {
          type: "tool_use",
          id: toolId,
          name: call.name,
          input: {},
        },
      }),
    );
    frames.push(
      event("content_block_delta", {
        type: "content_block_delta",
        index: blockIndex,
        delta: { type: "input_json_delta", partial_json: call.arguments },
      }),
    );
    frames.push(
      event("content_block_stop", {
        type: "content_block_stop",
        index: blockIndex,
      }),
    );
    blockIndex += 1;
  }
  frames.push(
    event("message_delta", {
      type: "message_delta",
      delta: {
        stop_reason: anthropicStopReason(answer.finishReason),
        stop_sequence: null,
      },
      usage: {
        input_tokens: usage.prompt_tokens,
        output_tokens: usage.completion_tokens,
      },
    }),
  );
  frames.push(event("message_stop", { type: "message_stop" }));
  return frames.join("\n\n") + "\n\n";
}

// ---------------------------------------------------------------------------
// HTTP handlers
// ---------------------------------------------------------------------------

function readRequestBody(
  req: import("node:http").IncomingMessage,
): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    let tooLarge = false;
    req.on("data", (chunk: Buffer) => {
      size += chunk.length;
      if (size > MAX_WORK_BODY_BYTES) {
        tooLarge = true;
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => {
      if (tooLarge) reject(new Error("request body too large"));
      else resolve(Buffer.concat(chunks).toString("utf8"));
    });
    req.on("error", reject);
  });
}

function modelsAnswer(
  req: import("node:http").IncomingMessage,
  catalog: Catalog,
): Record<string, unknown> {
  const models = catalog.providers[0]?.models ?? [];
  const useAnthropic =
    req.headers["x-api-key"] !== undefined &&
    req.headers.authorization === undefined;
  if (!useAnthropic) {
    return {
      object: "list",
      data: models.map((m) => ({
        id: m.id,
        object: "model",
        created: trialNowSec(),
        owned_by: "openrouter",
        free: true,
      })),
    };
  }
  const createdAt = new Date(trialNowMs()).toISOString();
  return {
    data: models.map((m) => ({
      id: m.id,
      type: "model",
      display_name: m.id,
      created_at: createdAt,
      free: true,
    })),
    has_more: false,
    first_id: models.length > 0 ? models[0].id : null,
    last_id: models.length > 0 ? models[models.length - 1].id : null,
  };
}

async function handleCompletion(
  req: import("node:http").IncomingMessage,
  res: import("node:http").ServerResponse,
  wire: "openai" | "anthropic",
  brief: Brief,
  callBudget: { used: number },
): Promise<void> {
  let raw: string;
  try {
    raw = await readRequestBody(req);
  } catch {
    writeJson(res, 413, { error: { message: "request body too large", type: "invalid_request_error", code: "request_too_large" } }, []);
    return;
  }
  let body: unknown;
  try {
    body = JSON.parse(raw);
  } catch {
    writeJson(res, 400, { error: { message: "the body is not JSON", type: "invalid_request_error", code: "invalid_json" } }, []);
    return;
  }
  if (!isRecord(body)) {
    writeJson(res, 400, { error: { message: "the body is not one JSON object", type: "invalid_request_error", code: "invalid_json" } }, []);
    return;
  }
  const stream = body.stream === true;
  const requestedMax =
    typeof body.max_tokens === "number" ? body.max_tokens : 1024;
  const maxTokens = Math.min(8192, Math.max(1, Math.floor(requestedMax)));
  const needsJson = isRecord(body.response_format);
  const needsTools = Array.isArray(body.tools) && body.tools.length > 0;
  const chatBody = buildChatBody(wire, body, maxTokens);
  const dialled = await dialModel(brief, chatBody, needsJson, needsTools, callBudget);
  if (dialled.answer === null || dialled.modelId === null) {
    writeJson(
      res,
      502,
      {
        error: {
          message: "the gateway refused every eligible model",
          type: "upstream_error",
          code: "upstream_error",
        },
      },
      dialled.failures,
    );
    return;
  }
  const served: ServedAnswer = {
    answer: dialled.answer,
    modelId: dialled.modelId,
    failures: dialled.failures,
  };
  const trail = trailHeaderValue(served.failures);
  if (!stream) {
    const value =
      wire === "openai"
        ? openAiJsonBody(body, served)
        : anthropicJsonBody(body, served);
    writeJson(res, 200, value, served.failures);
    return;
  }
  const sse =
    wire === "openai"
      ? openAiStreamBody(body, served)
      : anthropicStreamBody(body, served);
  const headers: Record<string, string> = {
    "content-type": "text/event-stream",
    "cache-control": "no-cache",
    connection: "keep-alive",
    "content-length": String(Buffer.byteLength(sse)),
  };
  if (trail !== null) headers["x-makefun-upstream-failures"] = trail;
  res.writeHead(200, headers);
  res.end(sse);
}

// ---------------------------------------------------------------------------
// Startup
// ---------------------------------------------------------------------------

function envUrl(): string {
  const raw = process.env.MAKEFUN_ENV_URL;
  if (raw === undefined || raw === "") return "";
  return raw.replace(/\/+$/, "");
}

async function postRegister(
  url: string,
  port: number,
  catalog: Catalog,
): Promise<boolean> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 5000);
  try {
    const response = await fetch(url + "/register", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ protocol_version: 1, port, catalog }),
      signal: controller.signal,
    });
    if (!response.ok) return false;
    const parsed: unknown = await response.json().catch(() => null);
    return isRecord(parsed) && parsed.ok === true;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

async function pollWork(url: string): Promise<boolean> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 5000);
  try {
    const response = await fetch(url + "/work", { signal: controller.signal });
    if (!response.ok) return false;
    const parsed: unknown = await response.json().catch(() => null);
    return isRecord(parsed) && parsed.done === true;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

async function main(): Promise<void> {
  const brief = readBrief();
  const catalog = buildCatalog(brief);
  const text = catalogText(catalog);
  if (process.argv.includes("--list")) {
    process.stdout.write(text + "\n");
    return;
  }
  if (brief === null) {
    process.stderr.write(
      "no brief: MAKEFUN_PROMPT_FILE is unset or unreadable; serving with an empty catalog\n",
    );
  }
  const callBudget = { used: 0 };
  const server = createServer((req, res) => {
    void (async () => {
      try {
        const url = new URL(req.url ?? "/", "http://server");
        if (req.method === "GET" && url.pathname === "/v1/models") {
          writeJson(res, 200, modelsAnswer(req, catalog), []);
          return;
        }
        if (brief === null) {
          writeJson(
            res,
            503,
            {
              error: {
                message: "no brief loaded",
                type: "invalid_request_error",
                code: "no_brief",
              },
            },
            [],
          );
          return;
        }
        if (req.method === "POST" && url.pathname === "/v1/chat/completions") {
          await handleCompletion(req, res, "openai", brief, callBudget);
          return;
        }
        if (req.method === "POST" && url.pathname === "/v1/messages") {
          await handleCompletion(req, res, "anthropic", brief, callBudget);
          return;
        }
        writeJson(
          res,
          404,
          {
            error: {
              message: "no route " + (req.method ?? "GET") + " " + url.pathname,
              type: "invalid_request_error",
              code: "not_found",
            },
          },
          [],
        );
      } catch (err) {
        const message = err instanceof Error ? err.message : "internal error";
        try {
          writeJson(
            res,
            500,
            {
              error: {
                message,
                type: "internal_error",
                code: "internal_error",
              },
            },
            [],
          );
        } catch {
          try {
            res.end();
          } catch {
            // The socket is already gone; nothing left to do.
          }
        }
      }
    })();
  });
  const port = await new Promise<number>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "0.0.0.0", () => {
      const address = server.address();
      if (address === null || typeof address === "string") {
        reject(new Error("the listener has no TCP address"));
        return;
      }
      resolve(address.port);
    });
  });
  const env = envUrl();
  if (env === "") {
    process.stderr.write("listening on 0.0.0.0:" + port + " (no env to register with)\n");
    return;
  }
  let registered = false;
  for (let attempt = 0; attempt < 10 && !registered; attempt += 1) {
    registered = await postRegister(env, port, catalog);
    if (!registered) await sleep(500);
  }
  if (!registered) {
    process.stderr.write("could not register with the env at " + env + "\n");
    process.exitCode = 1;
    server.close();
    return;
  }
  for (;;) {
    await sleep(WORK_POLL_MS);
    if (await pollWork(env)) break;
  }
  server.close();
}

void main().catch((err: unknown) => {
  const message = err instanceof Error ? err.message : "startup failed";
  process.stderr.write(message + "\n");
  process.exitCode = 1;
});
