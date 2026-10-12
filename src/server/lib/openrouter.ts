import { createDeepSeek } from "@ai-sdk/deepseek";
import {
  createOpenRouter,
  type LanguageModelV3,
} from "@openrouter/ai-sdk-provider";
import { defaultSettingsMiddleware, wrapLanguageModel } from "ai";
import { getOptionalEnvValue } from "@/server/lib/runtime-env";

// OpenRouter model slug used for the SAM in-app chat agent. Override with
// OPENROUTER_MODEL to swap models without a code change.
const DEFAULT_CHAT_AGENT_MODEL = "openai/gpt-5.6-luna";

// Previous default; kept reachable via OPENROUTER_MODEL for rollback. Its
// routing needs the ZDR/provider tuning below.
const MINIMAX_M3 = "minimax/minimax-m3";

/**
 * Returns the AI SDK LanguageModel for the chat agent. `usage: { include: true }`
 * turns on OpenRouter usage accounting so each response carries its real USD
 * cost (providerMetadata.openrouter.usage.cost) — which we meter against the
 * shared usage-credit pool.
 *
 * Default model: GPT-5.6 Luna at `reasoning.effort: "max"` — "max" is valid at
 * the OpenRouter API for GPT-5.x but missing from the SDK's effort union, so
 * the reasoning config rides in `extraBody`. Reasoning tokens stream on the
 * separate reasoning channel and are billed as output tokens, which the usage
 * accounting above captures.
 *
 * Sync on purpose: Think's `getModel()` hook is sync and runs on every turn,
 * so the SAM agent reads the key/model from its DO env and builds here.
 */
export function buildChatAgentModel(
  credentials: ChatAgentCredentials,
  reasoningEffort: "max" | "low" = "max",
): LanguageModelV3 {
  const { apiKey } = credentials;
  if (credentials.provider === "deepseek") {
    return wrapLanguageModel({
      model: createDeepSeek({ apiKey })(
        credentials.model ?? DEFAULT_DEEPSEEK_MODEL,
      ),
      middleware: defaultSettingsMiddleware({
        settings: { providerOptions: { deepseek: { reasoningEffort } } },
      }),
    });
  }
  const model = credentials.model ?? DEFAULT_CHAT_AGENT_MODEL;
  const openrouter = createOpenRouter({ apiKey });

  // MiniMax M3 (env-override path only): `provider.order` prefers Together,
  // then Atlas Cloud (fp8); `zdr: true` restricts routing to Zero-Data-
  // Retention endpoints, which excludes MiniMax first-party — the account's
  // "Non-frontier requires ZDR" data policy enforces the same, this flag is
  // belt-and-braces. Fallbacks stay on within the ZDR set because pinning
  // providers caused a prod outage (Jul 2026: Together upstream-rate-limited
  // m3 and every chat turn 429'd). The explicit reasoning channel keeps m3's
  // `<think>` trace out of the visible answer text.
  if (model === MINIMAX_M3) {
    return openrouter(model, {
      usage: { include: true },
      reasoning: { effort: reasoningEffort === "low" ? "low" : "medium" },
      provider: {
        order: ["together", "atlas-cloud/fp8"],
        zdr: true,
        allow_fallbacks: true,
      },
    });
  }

  return openrouter(model, {
    usage: { include: true },
    extraBody: { reasoning: { effort: reasoningEffort } },
  });
}

// DeepSeek's own API, used instead of OpenRouter when DEEPSEEK_API_KEY is set.
// Override the model with DEEPSEEK_MODEL (a DeepSeek API id, not an OpenRouter
// slug).
const DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash";

export type ChatAgentCredentials = {
  provider: "deepseek" | "openrouter";
  apiKey: string;
  model?: string;
};

/** DEEPSEEK_API_KEY wins over OPENROUTER_API_KEY; undefined when neither is set. */
export function chatAgentCredentials(
  get: (name: string) => string | undefined,
): ChatAgentCredentials | undefined {
  const deepseekKey = get("DEEPSEEK_API_KEY");
  if (deepseekKey) {
    return {
      provider: "deepseek",
      apiKey: deepseekKey,
      model: get("DEEPSEEK_MODEL"),
    };
  }
  const openrouterKey = get("OPENROUTER_API_KEY");
  return openrouterKey
    ? {
        provider: "openrouter",
        apiKey: openrouterKey,
        model: get("OPENROUTER_MODEL"),
      }
    : undefined;
}

export async function requireChatAgentCredentials(): Promise<ChatAgentCredentials> {
  const values = new Map<string, string | undefined>();
  for (const name of [
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_MODEL",
    "OPENROUTER_API_KEY",
    "OPENROUTER_MODEL",
  ]) {
    values.set(name, await getOptionalEnvValue(name));
  }
  return requireCredentials((name) => values.get(name));
}

export function requireCredentials(
  get: (name: string) => string | undefined,
): ChatAgentCredentials {
  const credentials = chatAgentCredentials(get);
  if (!credentials) {
    throw new Error(
      "Missing required environment variable: DEEPSEEK_API_KEY or OPENROUTER_API_KEY",
    );
  }
  return credentials;
}
