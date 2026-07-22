// 加载 .env 文件里的环境变量，比如 OPENAI_API_KEY、OPENAI_BASE_URL
import 'dotenv/config';

// 从 LangChain 引入 OpenAI 聊天模型封装
import { ChatOpenAI } from '@langchain/openai';

// 引入不同类型的消息：用户消息、系统消息、工具返回消息
import { HumanMessage, SystemMessage, ToolMessage } from '@langchain/core/messages';

// 引入我们自己定义的四个工具
import {
  executeCommandTool,
  listDirectoryTool,
  readFileTool,
  writeFileTool,
} from './all-tools.mjs';

// 引入 chalk，用来给终端输出加颜色
import chalk from 'chalk';

// 创建一个大模型实例
const model = new ChatOpenAI({
  // 使用的模型名称
  modelName: 'doubao-seed-2-0-mini-260428',

  // 从环境变量读取 API Key
  apiKey: process.env.OPENAI_API_KEY,

  // temperature 为 0，表示回答更稳定、更少随机性
  temperature: 0,

  // 自定义 OpenAI 兼容接口地址
  configuration: {
    // 从环境变量读取 baseURL
    baseURL: process.env.OPENAI_BASE_URL,
  },
});

// 定义 AI 可以使用的工具列表
const tools = [
  // 读取文件工具
  readFileTool,

  // 写入文件工具
  writeFileTool,

  // 执行命令工具
  executeCommandTool,

  // 列出目录工具
  listDirectoryTool,
];

// 把工具绑定到模型上，让模型可以主动调用这些工具
const modelWithTools = model.bindTools(tools);

// 定义 Agent 执行函数
async function runAgentWithTools(query, maxIterations = 30) {
  // messages 保存完整对话历史
  const messages = [
    // 系统消息：告诉 AI 它的身份、规则、工具用法
    new SystemMessage(`
你是一个项目管理助手，可以使用工具完成任务。

当前工作目录: ${process.cwd()}

你可以使用这些工具：
1. read_file: 读取文件
2. write_file: 写入文件
3. execute_command: 执行命令，支持 workingDirectory 参数
4. list_directory: 列出目录内容

重要规则 - execute_command:
- workingDirectory 参数会自动切换到指定目录
- 当使用 workingDirectory 时，不要在 command 里再写 cd
- 错误示例: { command: "cd react-todo-app && pnpm install", workingDirectory: "react-todo-app" }
- 正确示例: { command: "pnpm install", workingDirectory: "react-todo-app" }

重要规则 - write_file:
- 写 React 组件文件时，如果有对应 CSS 文件，记得在组件文件中 import CSS
`),

    // 用户消息：真正要交给 AI 完成的任务
    new HumanMessage(query),
  ];

  // 最多循环 maxIterations 次，防止 AI 无限调用工具
  for (let i = 0; i < maxIterations; i++) {
    // 在终端打印提示，表示正在等待 AI 回复
    console.log(chalk.bgGreen(`正在等待 AI 思考...`));

    // 把当前所有消息发给模型，获得 AI 回复
    const response = await modelWithTools.invoke(messages);

    // 把 AI 回复加入对话历史
    messages.push(response);

    // 如果 AI 没有调用工具，说明它已经给出最终答案
    if (!response.tool_calls || response.tool_calls.length === 0) {
      // 打印最终回答
      console.log(`\nAI 最终回答:\n${response.content}\n`);

      // 返回最终回答
      return response.content;
    }

    // 如果 AI 调用了工具，就逐个执行工具调用
    for (const toolCall of response.tool_calls) {
      // 根据 AI 提供的工具名，找到对应工具
      const foundTool = tools.find((tool) => tool.name === toolCall.name);

      // 如果找到了工具，就执行它
      if (foundTool) {
        // 把 AI 给出的参数传给工具，并等待工具执行结果
        const toolResult = await foundTool.invoke(toolCall.args);

        // 把工具结果包装成 ToolMessage，放回对话历史
        messages.push(
          new ToolMessage({
            // 工具返回的内容
            content: toolResult,

            // 工具调用 ID，用来对应这次 tool call
            tool_call_id: toolCall.id,
          })
        );
      }
    }
  }

  // 如果循环次数用完，就返回最后一条消息内容
  return messages[messages.length - 1].content;
}

// 定义一个测试任务，让 AI 创建 React TodoList 项目
const case1 = `
创建一个功能丰富的 React TodoList 应用：

1. 创建项目：
   pnpm create vite react-todo-app --template react-ts

2. 修改 src/App.tsx，实现完整功能的 TodoList：
   - 添加、删除、编辑、标记完成
   - 分类筛选：全部 / 进行中 / 已完成
   - 统计信息显示
   - localStorage 数据持久化

3. 添加复杂样式：
   - 渐变背景
   - 卡片阴影、圆角
   - 悬停效果

4. 添加动画：
   - 添加 / 删除时的过渡动画
   - 使用 CSS transitions

5. 列出目录确认

注意：
- 使用 pnpm
- 功能要完整
- 样式要美观
- 要有动画效果

之后在 react-todo-app 项目中：
1. 使用 pnpm install 安装依赖
2. 使用 pnpm run dev 启动服务器
`;

// 捕获整个 Agent 执行过程中的错误
try {
  // 启动 Agent，把 case1 任务交给 AI
  await runAgentWithTools(case1);
} catch (error) {
  // 如果出错，就打印错误信息
  console.error(`\n错误: ${error.message}\n`);
}