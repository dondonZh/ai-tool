// 自动读取项目根目录下的 .env 文件，并把其中的配置放进 process.env。
// 这里没有接收任何变量，因为只要“导入”这个模块，它就会自动完成环境变量加载。
import 'dotenv/config';

// ChatOpenAI 是 LangChain 对“兼容 OpenAI 接口的聊天模型”的一层封装。
// 虽然下面实际使用的是 qwen-plus，但它提供了 OpenAI 兼容接口，所以也能用这个类。
import { ChatOpenAI } from '@langchain/openai';

// HumanMessage：代表用户发给 AI 的消息。
// SystemMessage：代表给 AI 设定身份、能力和规则的系统消息。
// ToolMessage：代表某个工具执行完以后，返回给 AI 的结果消息。
import { HumanMessage, SystemMessage, ToolMessage } from '@langchain/core/messages';

// InMemoryChatMessageHistory 用来把本次 Agent 运行中的所有消息保存在内存里。
// 有了它，模型在下一轮思考时才能看到前面发生过什么。
import { InMemoryChatMessageHistory } from '@langchain/core/chat_history';

// 这个解析器会尝试把模型流式输出的“工具调用 JSON 碎片”拼成可读的工具调用。
// 本文件主要用它提前预览 write_file 正在生成的文件内容。
import { JsonOutputToolsParser } from '@langchain/core/output_parsers/openai_tools';

// 导入 Agent 可以使用的四个本地工具：执行命令、列目录、读文件、写文件。
// 这些工具的具体实现和参数定义都在同目录的 all-tools.mjs 中。
import { executeCommandTool, listDirectoryTool, readFileTool, writeFileTool } from './all-tools.mjs';

// chalk 用来给终端文字加颜色，只影响显示效果，不影响 Agent 逻辑。
import chalk from 'chalk';

// 创建一个聊天模型实例；后面的 Agent 每一轮都会通过它向大模型发请求。
const model = new ChatOpenAI({
    // 指定真正调用的模型名称。
    modelName: 'qwen-plus',

    // 从 .env 读取 API 密钥，避免把密钥直接写死在代码里。
    apiKey: process.env.OPENAI_API_KEY,

    // 温度设为 0，让模型回答更稳定、随机性更低，更适合执行代码任务。
    temperature: 0,

    // configuration 用来传递底层 OpenAI 客户端的额外配置。
    configuration: {
        // 使用 .env 中的接口地址，因此这里可以连接任何兼容 OpenAI 协议的服务。
        baseURL: process.env.OPENAI_BASE_URL,
    },
});

// 把所有允许模型调用的工具集中放进一个数组，方便统一绑定、查找和执行。
const tools = [
    // 让 Agent 可以读取文件内容。
    readFileTool,

    // 让 Agent 可以创建或覆盖文件。
    writeFileTool,

    // 让 Agent 可以执行 pnpm、npm 等系统命令。
    executeCommandTool,

    // 让 Agent 可以查看某个目录里有哪些文件和文件夹。
    listDirectoryTool,
];

// 把上面的工具说明绑定到模型。
// 绑定后，模型不只会回复文字，还能返回“我要调用哪个工具、参数是什么”。
const modelWithTools = model.bindTools(tools);

/**
 * 运行一个能够反复调用工具的 AI Agent。
 *
 * 最核心的循环是：
 * 用户任务 -> 模型思考 -> 模型要求调用工具 -> 程序执行工具
 *          -> 把工具结果交还模型 -> 模型继续思考 -> 最终回答
 *
 * @param {string} query 用户希望 Agent 完成的任务。
 * @param {number} maxIterations 最多允许模型思考多少轮，防止 Agent 无限循环。
 * @returns {Promise<unknown>} Agent 的最终回复内容。
 */
async function runAgentWithTools(query, maxIterations = 30) {
    // 新建一份只存在于内存中的对话历史。
    // 每调用一次本函数，都会得到一份全新的历史，互相不会串话。
    const history = new InMemoryChatMessageHistory();

    // 先放入系统消息，告诉模型“你是谁、有哪些工具、使用工具时要遵守什么规则”。
    await history.addMessage(
        // SystemMessage 的优先级高于普通用户消息，适合放长期规则。
        new SystemMessage(`你是一个项目管理助手，使用工具完成任务。

当前工作目录: ${process.cwd()}

工具：
1. read_file: 读取文件
2. write_file: 写入文件
3. execute_command: 执行命令（支持 workingDirectory 参数）
4. list_directory: 列出目录

重要规则 - execute_command：
- workingDirectory 参数会自动切换到指定目录
- 当使用 workingDirectory 时，绝对不要在 command 中使用 cd
- 错误示例: { command: "cd react-todo-app && pnpm install", workingDirectory: "react-todo-app" }
- 正确示例: { command: "pnpm install", workingDirectory: "react-todo-app" }

重要规则 - write_file：
- 当写入 React 组件文件（如 App.tsx）时，如果存在对应的 CSS 文件（如 App.css），在其他 import 语句后加上这个 css 的导入
`),
    );

    // 再把函数收到的 query 包装成“用户消息”，追加到对话历史中。
    await history.addMessage(new HumanMessage(query));

    // Agent 可能需要多次调用工具，所以用循环让模型最多工作 maxIterations 轮。
    for (let i = 0; i < maxIterations; i++) {
        // 在终端显示绿色提示，让人知道程序正在等待模型响应。
        console.log(chalk.bgGreen('⏳ 正在等待 AI 思考...'));

        // 取出截至目前的完整消息历史，包括系统消息、用户消息、AI 消息和工具结果。
        const messages = await history.getMessages();

        // 把完整历史发给模型，并要求模型用“流式”方式一点点返回结果。
        // 这里先拿到的是可异步遍历的流，而不是一次性返回的完整 AI 消息。
        const rawStream = await modelWithTools.stream(messages);

        // 这个变量最终会保存本轮完整的 AIMessage。
        // 初始值是 null，因为此刻还没有收到任何输出片段。
        let fullAIMessage = null;

        // 创建工具调用解析器，尝试从尚未结束的输出流里提前解析工具参数。
        const toolParser = new JsonOutputToolsParser();

        // Map 用来记录每个 write_file 工具调用已经打印了多少个字符。
        // 这样下一次收到更完整的内容时，只打印新增长的部分，不会重复刷屏。
        const printedLengths = new Map();

        // 在终端显示蓝色提示，表示马上开始接收和展示模型的流式输出。
        console.log(chalk.bgBlue('\n🚀 Agent 开始思考并生成流...\n'));

        // for await...of 会等待并逐个取出模型返回的 AIMessageChunk（消息小片段）。
        for await (const chunk of rawStream) {
            // 第一块到来时，直接把 chunk 当作完整消息的起点。
            // 后续块到来时，用 concat 把新片段接到已有消息后面。
            fullAIMessage = fullAIMessage ? fullAIMessage.concat(chunk) : chunk;

            // 默认认为当前还没有成功解析出工具调用。
            let parsedTools = null;

            // 流式 JSON 在输出结束前经常是不完整的，所以解析操作必须允许失败。
            try {
                // 把目前累计出的完整消息交给解析器，尝试得到工具调用数组。
                // parseResult 的输入格式要求是“候选生成结果数组”，所以外面要包一层数组和 message 对象。
                parsedTools = await toolParser.parseResult([{ message: fullAIMessage }]);
            } catch (error) {
                // 解析失败通常只是因为工具参数的 JSON 还没有输出完，并不代表程序真的出错。
                // 因此这里故意不抛出错误，让循环继续等待下一块内容。
            }

            // 只有确实解析出了至少一个工具调用时，才进入工具调用预览逻辑。
            if (parsedTools && parsedTools.length > 0) {
                // 一次模型回复可能要求调用多个工具，所以逐个检查。
                for (const toolCall of parsedTools) {
                    // 本程序只对 write_file 做流式预览，而且必须已经解析到 content 参数。
                    if (toolCall.type === 'write_file' && toolCall.args?.content) {
                        // 优先用工具调用的唯一 id 区分不同调用。
                        // 如果暂时还没有 id，就依次退回到文件路径和固定默认值。
                        const toolCallId = toolCall.id || toolCall.args.filePath || 'default';

                        // 确保 content 是字符串，方便使用 length 和 slice 计算新增内容。
                        const currentContent = String(toolCall.args.content);

                        // 查出这个工具调用上一次已经打印到第几个字符。
                        // 如果从未打印过，Map.get 会返回 undefined。
                        const previousLength = printedLengths.get(toolCallId);

                        // undefined 表示这是第一次看到这个 write_file 工具调用。
                        if (previousLength === undefined) {
                            // 先登记为已经打印了 0 个字符。
                            printedLengths.set(toolCallId, 0);

                            // 打印一次蓝色标题，告诉用户 Agent 准备写哪个文件。
                            console.log(
                                // chalk.bgBlue 给这段标题加上蓝色背景。
                                chalk.bgBlue(
                                    // filePath 来自模型生成的 write_file 工具参数。
                                    `\n[工具调用] write_file("${toolCall.args.filePath}") - 开始写入（流式预览）\n`,
                                ),
                            );
                        }

                        // 只有当前内容比上次更长，才说明模型又生成了新的文件内容。
                        if (currentContent.length > previousLength) {
                            // 从旧长度处切片，只取这一次新增加的字符串。
                            const newContent = currentContent.slice(previousLength);

                            // 用 stdout.write 直接输出，不自动换行，才能得到连续的“打字机”效果。
                            process.stdout.write(newContent);

                            // 更新已打印长度，下一次就会从当前末尾继续打印。
                            printedLengths.set(toolCallId, currentContent.length);
                        }
                    }
                }
            } else {
                // 没解析到工具调用，说明当前输出更可能是模型直接回复的普通文字。

                // chunk.content 有内容时才输出，避免打印空值。
                if (chunk.content) {
                    // 把当前文字片段直接写到终端，保留流式输出效果。
                    process.stdout.write(
                        // 大多数模型返回字符串；如果返回的是结构化内容，就转成 JSON 字符串再显示。
                        typeof chunk.content === 'string'
                            ? chunk.content
                            : JSON.stringify(chunk.content),
                    );
                }
            }
        }

        // 走到这里说明本轮模型输出流已经结束，fullAIMessage 已拼成一条完整 AI 消息。
        // 把它存入历史，下一轮模型才能记得自己刚才说过什么、请求过哪些工具。
        await history.addMessage(fullAIMessage);

        // 打印绿色状态，方便观察对话历史保存到了哪个阶段。
        console.log(chalk.green('\n✅ 消息已完整存入历史'));

        // 如果完整消息里没有任何工具调用，说明模型认为任务已经完成。
        if (!fullAIMessage.tool_calls || fullAIMessage.tool_calls.length === 0) {
            // 在终端展示模型的最终文字回复。
            console.log(`\n✨ AI 最终回复:\n${fullAIMessage.content}\n`);

            // 返回最终回复，并立刻结束整个 Agent 函数。
            return fullAIMessage.content;
        }

        // 如果模型要求调用工具，就逐个执行本轮消息中的工具调用。
        for (const toolCall of fullAIMessage.tool_calls) {
            // 根据模型给出的工具名，在本地 tools 数组里找到真正的工具对象。
            const foundTool = tools.find((toolItem) => toolItem.name === toolCall.name);

            // 只有本地确实存在这个工具时才执行，避免对 undefined 调用 invoke。
            if (foundTool) {
                // 用模型生成的参数执行工具，并等待读取、写入或命令执行完成。
                const toolResult = await foundTool.invoke(toolCall.args);

                // 把工具执行结果包装成 ToolMessage，再追加到对话历史。
                await history.addMessage(
                    // ToolMessage 告诉模型：“这是你刚才请求的那个工具的返回结果”。
                    new ToolMessage({
                        // content 是工具真正返回给模型阅读的内容。
                        content: toolResult,

                        // tool_call_id 把结果和模型原先的那次工具请求准确对应起来。
                        tool_call_id: toolCall.id,
                    }),
                );
            }
        }

        // 工具结果写入历史后，本轮结束；for 循环会开始下一轮模型思考。
    }

    // 只有连续运行到最大轮数、始终没有正常 return 时，才会执行到这里。
    // 先取出停止时的全部历史消息。
    const finalMessages = await history.getMessages();

    // 返回最后一条消息的内容，让调用者至少能拿到 Agent 停止前的最新结果。
    return finalMessages[finalMessages.length - 1].content;
}

// 这是交给 Agent 的示例任务。
// 使用模板字符串可以方便地写多行文字，并完整保留换行和任务步骤。
const case1 = `创建一个功能丰富的 React TodoList 应用：

1. 创建项目：echo -e "n\nn" | pnpm create vite react-todo-app --template react-ts
2. 修改 src/App.tsx，实现完整功能的 TodoList：
 - 添加、删除、编辑、标记完成
 - 分类筛选（全部/进行中/已完成）
 - 统计信息显示
 - localStorage 数据持久化
3. 添加复杂样式：
 - 渐变背景（蓝到紫）
 - 卡片阴影、圆角
 - 悬停效果
4. 添加动画：
 - 添加/删除时的过渡动画
 - 使用 CSS transitions
5. 列出目录确认

注意：使用 pnpm，功能要完整，样式要美观，要有动画效果

去掉 main.tsx 里的 index.css 导入

之后在 react-todo-app 项目中：
1. 使用 pnpm install 安装依赖
2. 使用 pnpm run dev 启动服务器
`;

// 用 try...catch 包住 Agent 主入口，避免异常没有处理就直接让 Node.js 崩溃。
try {
    // 启动 Agent，并等待它完成上面 case1 描述的任务。
    await runAgentWithTools(case1);
} catch (error) {
    // 如果模型请求、工具执行或其他步骤抛出异常，就在终端打印错误原因。
    console.error(`\n❌ 错误: ${error.message}\n`);
}
