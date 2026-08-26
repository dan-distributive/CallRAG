// Provider-agnostic chat interface. Set LLM_PROVIDER=anthropic (default) or
// LLM_PROVIDER=local (any OpenAI chat-completions-compatible server --
// Ollama, vLLM, llama.cpp's server mode, and LM Studio all speak this
// format out of the box) via env var. Callers (queryEngine.js) never see
// the difference -- both paths return the same {stopReason, content,
// usage} shape, content being an array of {type:'text', text} and/or
// {type:'tool_use', id, name, input} blocks, matching Anthropic's own
// message content shape since that's the richer of the two and the
// OpenAI shape normalizes into it losslessly.
//
// The 'local' path is structurally complete but UNVERIFIED end-to-end --
// no local LLM server was available in the environment this was built in.
// It targets the documented OpenAI chat-completions request/response
// shape, which Ollama/vLLM/llama.cpp-server/LM Studio all implement, so it
// should work as soon as one is pointed at via LLM_BASE_URL, but "should"
// is not "confirmed."

const PROVIDER = process.env.LLM_PROVIDER || 'anthropic';

/**
 * @param {Object} params
 * @param {string} [params.system] system prompt
 * @param {Array<{role: string, content: any}>} params.messages
 * @param {Array<{name: string, description: string, input_schema: object}>} [params.tools]
 * @param {string} [params.model] overrides the provider's default model
 * @returns {Promise<{stopReason: string, content: Array, usage: {input_tokens: number, output_tokens: number}}|{error: string}>}
 */
async function chat({ system, messages, tools, model }) {
  return PROVIDER === 'local' ? chatLocal({ system, messages, tools, model }) : chatAnthropic({ system, messages, tools, model });
}

async function chatAnthropic({ system, messages, tools, model }) {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) return { error: 'ANTHROPIC_API_KEY not set' };

  let res;
  try {
    res = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-api-key': apiKey, 'anthropic-version': '2023-06-01' },
      body: JSON.stringify({
        model: model || process.env.LLM_MODEL || 'claude-sonnet-5',
        max_tokens: 1536,
        system,
        messages,
        tools,
      }),
    });
  } catch (err) {
    return { error: `Anthropic request failed: ${err.message}` };
  }
  const data = await res.json();
  if (!res.ok) return { error: `Anthropic API error: ${JSON.stringify(data)}` };

  return {
    stopReason: data.stop_reason,
    content: data.content,
    usage: { input_tokens: data.usage?.input_tokens ?? 0, output_tokens: data.usage?.output_tokens ?? 0 },
  };
}

async function chatLocal({ system, messages, tools, model }) {
  const baseUrl = process.env.LLM_BASE_URL || 'http://localhost:11434/v1';

  // Anthropic messages can have array-of-blocks content (text, tool_use,
  // tool_result); OpenAI's chat-completions wants plain strings for
  // user/assistant/system and a separate tool_calls field on assistant
  // messages, plus role:'tool' for results -- translate both directions.
  const openaiMessages = [];
  if (system) openaiMessages.push({ role: 'system', content: system });
  for (const m of messages) {
    if (typeof m.content === 'string') {
      openaiMessages.push({ role: m.role, content: m.content });
      continue;
    }
    const toolResults = m.content.filter((c) => c.type === 'tool_result');
    const toolUses = m.content.filter((c) => c.type === 'tool_use');
    const text = m.content.filter((c) => c.type === 'text').map((c) => c.text).join('');
    if (toolResults.length) {
      for (const tr of toolResults) openaiMessages.push({ role: 'tool', tool_call_id: tr.tool_use_id, content: tr.content });
      continue;
    }
    openaiMessages.push({
      role: m.role,
      content: text || null,
      ...(toolUses.length
        ? { tool_calls: toolUses.map((tu) => ({ id: tu.id, type: 'function', function: { name: tu.name, arguments: JSON.stringify(tu.input) } })) }
        : {}),
    });
  }

  const openaiTools = tools
    ? tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.input_schema } }))
    : undefined;

  let res;
  try {
    res = await fetch(`${baseUrl}/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model: model || process.env.LLM_MODEL || 'llama3.1', messages: openaiMessages, tools: openaiTools }),
    });
  } catch (err) {
    return { error: `Local LLM request failed (is ${baseUrl} reachable?): ${err.message}` };
  }
  const data = await res.json();
  if (!res.ok) return { error: `Local LLM error: ${JSON.stringify(data)}` };

  const choice = data.choices[0];
  const content = [];
  if (choice.message.content) content.push({ type: 'text', text: choice.message.content });
  for (const tc of choice.message.tool_calls || []) {
    content.push({ type: 'tool_use', id: tc.id, name: tc.function.name, input: JSON.parse(tc.function.arguments) });
  }

  return {
    stopReason: choice.finish_reason === 'tool_calls' ? 'tool_use' : 'end_turn',
    content,
    usage: { input_tokens: data.usage?.prompt_tokens ?? 0, output_tokens: data.usage?.completion_tokens ?? 0 },
  };
}

module.exports = { chat };
