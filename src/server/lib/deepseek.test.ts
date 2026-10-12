import { describe, expect, it } from "vitest";
import { chatStepCostUsd, requireChatStepCostUsd } from "./chatAgent";
import { chatAgentCredentials } from "./openrouter";

const env = (values: Record<string, string>) => (name: string) => values[name];

describe("chatAgentCredentials", () => {
  it("prefers DeepSeek over OpenRouter and keeps each provider's model", () => {
    expect(
      chatAgentCredentials(
        env({
          DEEPSEEK_API_KEY: "ds",
          DEEPSEEK_MODEL: "deepseek-v4-pro",
          OPENROUTER_API_KEY: "or",
          OPENROUTER_MODEL: "openai/gpt-5.6-luna",
        }),
      ),
    ).toEqual({ provider: "deepseek", apiKey: "ds", model: "deepseek-v4-pro" });
    expect(chatAgentCredentials(env({ OPENROUTER_API_KEY: "or" }))).toEqual({
      provider: "openrouter",
      apiKey: "or",
      model: undefined,
    });
    expect(chatAgentCredentials(env({}))).toBeUndefined();
  });
});

describe("chat step cost", () => {
  const usage = {
    inputTokens: 1_000_000,
    outputTokens: 1_000_000,
    inputTokenDetails: { cacheReadTokens: 500_000 },
  };

  it("prices DeepSeek steps from tokens, splitting cache hits", () => {
    // 0.5M miss * 0.14 + 0.5M hit * 0.0028 + 1M out * 0.28
    expect(
      requireChatStepCostUsd({ providerMetadata: { deepseek: {} }, usage }),
    ).toBeCloseTo(0.07 + 0.0014 + 0.28, 6);
  });

  it("keeps OpenRouter's reported cost and strictness", () => {
    const providerMetadata = { openrouter: { usage: { cost: 0.42 } } };
    expect(chatStepCostUsd({ providerMetadata, usage })).toBe(0.42);
    expect(chatStepCostUsd({ providerMetadata: {}, usage })).toBe(0);
    expect(() =>
      requireChatStepCostUsd({ providerMetadata: {}, usage }),
    ).toThrow();
  });
});
