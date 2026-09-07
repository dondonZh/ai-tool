/*
 * @Author: dali crayoncrook@gmail.com
 * @Date: 2026-09-07 16:31:46
 * @LastEditors: dali crayoncrook@gmail.com
 * @LastEditTime: 2026-09-07 16:32:55
 * @FilePath: \tool-test\src\structured-json-schema.mjs
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
import 'dotenv/config';
import { ChatOpenAI } from '@langchain/openai';
import chalk from 'chalk';
import { z } from 'zod';
import { zodToJsonSchema } from "zod-to-json-schema";
import { HumanMessage, SystemMessage } from '@langchain/core/messages';

const scientistSchema = z.object({
    name: z.string().describe("科学家的全名"),
    birth_year: z.number().describe("出生年份"),
    field: z.string().describe("主要研究领域"),
    achievements: z.array(z.string()).describe("主要成就列表")
}).strict();

// 将 Zod 转换为原生的 JSON Schema 格式
const nativeJsonSchema = zodToJsonSchema(scientistSchema);

const model = new ChatOpenAI({
    modelName: "qwen-max",
    temperature: 0,
    apiKey: process.env.OPENAI_API_KEY,
    configuration: {
        baseURL: process.env.OPENAI_BASE_URL,
    },
    modelKwargs: { // 通过 modelKwargs 传入原生参数
        response_format: {
            type: "json_schema",
            json_schema: {
                name: "scientist_info",
                strict: true,
                schema: nativeJsonSchema // 这里的 nativeJsonSchema 就是转换后的对象
            }
        }
    }
});

async function testNativeJsonSchema() {
    console.log(chalk.bgMagenta("🧪 测试原生 JSON Schema 模式...\n"));

    const res = await model.invoke([
        new SystemMessage("你是一个信息提取助手，请直接返回 JSON 数据。"),
        new HumanMessage("介绍一下杨振宁")
    ]);

    console.log(chalk.green("\n✅ 收到响应 (纯净 JSON):"));
    console.log(res.content); 

    const data = JSON.parse(res.content);
    console.log(chalk.cyan("\n📋 解析后的对象:"));
    console.log(data);
}

testNativeJsonSchema().catch(console.error);
