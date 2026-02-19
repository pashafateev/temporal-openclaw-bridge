import { readFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

import express from "express";
import {
  Client,
  Connection,
  ConnectionOptions,
  WorkflowHandle,
  WorkflowUpdateStage,
} from "@temporalio/client";

type ConversationItem = {
  type: string;
  seq: number;
  content?: string;
  turn_id?: string;
};

type SessionConfiguration = {
  base_instructions?: string;
  user_instructions?: string;
  model: {
    provider: string;
    model: string;
    temperature: number;
    max_tokens: number;
    context_window: number;
  };
  tools: {
    enabled_tools: string[];
  };
  approval_mode?: string;
  cwd?: string;
  session_source?: string;
};

type WorkflowInput = {
  conversation_id: string;
  user_message: string;
  config: SessionConfiguration;
};

const TEMPORAL_HOST = process.env.TEMPORAL_HOST ?? "localhost:7233";
const TEMPORAL_NAMESPACE = process.env.TEMPORAL_NAMESPACE ?? "default";
const TASK_QUEUE = process.env.TEMPORAL_TASK_QUEUE ?? "temporal-agent-harness";
const WORKFLOW_TYPE = process.env.TEMPORAL_WORKFLOW_TYPE ?? "AgenticWorkflow";
const BRIDGE_PORT = Number(process.env.BRIDGE_PORT ?? 3001);
const WORKSPACE_ROOT = process.env.OPENCLAW_WORKSPACE_ROOT ?? "/Users/admin/.openclaw/workspace";
const CWD = process.env.AGENT_CWD ?? WORKSPACE_ROOT;
const MODEL_PROVIDER = process.env.MODEL_PROVIDER ?? detectDefaultProvider();
const MODEL_NAME =
  process.env.MODEL_NAME ??
  (MODEL_PROVIDER === "anthropic" ? "claude-sonnet-4-20250514" : "gpt-4o-mini");

function detectDefaultProvider(): string {
  if (process.env.ANTHROPIC_API_KEY) {
    return "anthropic";
  }
  return "openai";
}

async function buildSystemPrompt(): Promise<string> {
  const files = ["SOUL.md", "USER.md", "AGENTS.md", "MEMORY.md"];
  const parts: string[] = [];

  for (const file of files) {
    const filePath = path.join(WORKSPACE_ROOT, file);
    try {
      const content = await readFile(filePath, "utf8");
      parts.push(`# ${file}\n${content}`);
    } catch {
      // Missing files are acceptable in tracer mode.
    }
  }

  return parts.join("\n\n");
}

async function createTemporalClient(): Promise<Client> {
  const options: ConnectionOptions = {
    address: TEMPORAL_HOST,
  };
  const connection = await Connection.connect(options);
  return new Client({ connection, namespace: TEMPORAL_NAMESPACE });
}

function workflowIdForSession(sessionId: string): string {
  return `openclaw-${sessionId}`;
}

async function ensureWorkflow(
  client: Client,
  sessionId: string,
  initialMessage: string,
): Promise<WorkflowHandle> {
  const workflowId = workflowIdForSession(sessionId);
  const existing = client.workflow.getHandle(workflowId);
  try {
    await existing.describe();
    return existing;
  } catch {
    const baseInstructions = await buildSystemPrompt();
    const input: WorkflowInput = {
      conversation_id: workflowId,
      user_message: initialMessage,
      config: {
        base_instructions: baseInstructions,
        user_instructions: "OpenClaw bridge tracer mode",
        model: {
          provider: MODEL_PROVIDER,
          model: MODEL_NAME,
          temperature: 0.7,
          max_tokens: 4096,
          context_window: 128000,
        },
        tools: {
          enabled_tools: ["request_user_input"],
        },
        approval_mode: "never",
        cwd: CWD,
        session_source: "openclaw-bridge",
      },
    };

    return client.workflow.start(WORKFLOW_TYPE, {
      args: [input],
      taskQueue: TASK_QUEUE,
      workflowId,
    });
  }
}

async function sendUserInput(
  handle: WorkflowHandle,
  message: string,
): Promise<{ turn_id: string }> {
  const result = await handle.executeUpdate("user_input", {
    args: [{ content: message }],
    waitForStage: WorkflowUpdateStage.COMPLETED,
  });
  return result as { turn_id: string };
}

async function queryConversationItems(handle: WorkflowHandle): Promise<ConversationItem[]> {
  const items = await handle.query("get_conversation_items");
  return items as ConversationItem[];
}

async function waitForAssistantReply(
  handle: WorkflowHandle,
  sinceSeq: number,
  timeoutMs = 60_000,
): Promise<ConversationItem | null> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const items = await queryConversationItems(handle);
    const reply = items.find(
      (item) => item.seq > sinceSeq && item.type === "assistant_message" && item.content,
    );
    if (reply) {
      return reply;
    }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  return null;
}

async function startBridgeServer(): Promise<void> {
  const client = await createTemporalClient();
  const app = express();
  app.use(express.json());

  app.get("/healthz", (_req, res) => {
    res.status(200).json({ ok: true });
  });

  app.post("/session/start", async (req, res) => {
    const { sessionId, message } = req.body as { sessionId?: string; message?: string };
    if (!sessionId || !message) {
      res.status(400).json({ error: "sessionId and message are required" });
      return;
    }

    try {
      const handle = await ensureWorkflow(client, sessionId, message);
      res.json({ workflowId: handle.workflowId });
    } catch (error) {
      res.status(500).json({ error: String(error) });
    }
  });

  app.post("/session/send", async (req, res) => {
    const { sessionId, message } = req.body as { sessionId?: string; message?: string };
    if (!sessionId || !message) {
      res.status(400).json({ error: "sessionId and message are required" });
      return;
    }

    const handle = await ensureWorkflow(client, sessionId, message);
    try {
      const itemsBefore = await queryConversationItems(handle);
      const lastSeq = itemsBefore.length ? itemsBefore[itemsBefore.length - 1].seq : 0;
      const accepted = await sendUserInput(handle, message);
      const reply = await waitForAssistantReply(handle, lastSeq);
      if (reply?.content) {
        console.log(
          `[tracer] session=${sessionId} turn=${accepted.turn_id} assistant_reply=${reply.content}`,
        );
      }
      res.json({
        turnId: accepted.turn_id,
        reply: reply?.content ?? null,
      });
    } catch (error) {
      res.status(500).json({ error: String(error) });
    }
  });

  app.post("/reply", (req, res) => {
    const { sessionId, text } = req.body as { sessionId?: string; text?: string };
    if (!sessionId || !text) {
      res.status(400).json({ error: "sessionId and text are required" });
      return;
    }

    // Tracer phase: log instead of forwarding to Telegram/OpenClaw messaging layer.
    console.log(`[reply-hook] session=${sessionId} text=${text}`);
    res.json({ ok: true });
  });

  app.listen(BRIDGE_PORT, () => {
    console.log(`bridge listening on http://localhost:${BRIDGE_PORT}`);
    console.log(`temporal=${TEMPORAL_HOST} namespace=${TEMPORAL_NAMESPACE} taskQueue=${TASK_QUEUE}`);
    console.log(`model=${MODEL_PROVIDER}/${MODEL_NAME}`);
  });
}

startBridgeServer().catch((error) => {
  console.error("bridge failed to start", error);
  process.exit(1);
});
