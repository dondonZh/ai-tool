/*
 * @Author: dali crayoncrook@gmail.com
 * @Date: 2026-07-24 14:39:49
 * @LastEditors: dali crayoncrook@gmail.com
 * @LastEditTime: 2026-07-24 16:39:06
 * @FilePath: \tool-test\src\langchain-mcp-test.mjs
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
/*
 * @Author: dali crayoncrook@gmail.com
 * @Date: 2026-07-24 14:39:49
 * @LastEditors: dali crayoncrook@gmail.com
 * @LastEditTime: 2026-07-24 14:40:05
 * @FilePath: \tool-test\src\langchain-mcp-test.mjs
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
import 'dotenv/config';
import { MultiServerMCPClient } from '@langchain/mcp-adapters';
import { ChatOpenAI } from '@langchain/openai';
import chalk from 'chalk';
import { HumanMessage, SystemMessage, ToolMessage } from '@langchain/core/messages';

const model = new ChatOpenAI({ 
    modelName: "doubao-seed-2-0-mini-260428",
    apiKey: process.env.OPENAI_API_KEY,
    configuration: {
        baseURL: process.env.OPENAI_BASE_URL,
    },
});

const mcpClient = new MultiServerMCPClient({
    mcpServers: {
        'my-mcp-server': {
            command: "node",
            args: [
                "D:/vsproject/tool-test/src/my-mcp-server.mjs"
            ]
        }
    }
});

const tools = await mcpClient.getTools();
const modelWithTools = model.bindTools(tools);

const res = await mcpClient.listResources();

let resourceContent = '';
for (const [serverName, resources] of Object.entries(res)) {
    for (const resource of resources) {
        const content = await mcpClient.readResource(serverName, resource.uri);
        resourceContent += content[0].text;
    }
}

async function runAgentWithTools(query, maxIterations = 30) {
    const messages = [
        new SystemMessage(resourceContent),
        new HumanMessage(query)
    ];

    for (let i = 0; i < maxIterations; i++) {
        console.log(chalk.bgGreen(`⏳ 正在等待 AI 思考...`));
        const response = await modelWithTools.invoke(messages);
        messages.push(response);

        // 检查是否有工具调用
        if (!response.tool_calls || response.tool_calls.length === 0) {
            console.log(`\n✨ AI 最终回复:\n${response.content}\n`);
            return response.content;
        }

        console.log(chalk.bgBlue(`🔍 检测到 ${response.tool_calls.length} 个工具调用`));
        console.log(chalk.bgBlue(`🔍 工具调用: ${response.tool_calls.map(t => t.name).join(', ')}`));
        // 执行工具调用
        for (const toolCall of response.tool_calls) {
            const foundTool = tools.find(t => t.name === toolCall.name);
            if (foundTool) {
                const toolResult = await foundTool.invoke(toolCall.args);
                messages.push(new ToolMessage({
                    content: toolResult,
                    tool_call_id: toolCall.id,
                }));
            }
        }
    }

    return messages[messages.length - 1].content;
}


await runAgentWithTools("查一下用户 002 的信息");
// await runAgentWithTools("MCP Server 的使用指南是什么");

await mcpClient.close();