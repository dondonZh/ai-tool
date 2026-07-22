import dotenv from 'dotenv';
import { ChatOpenAI } from '@langchain/openai';

// 加载 .env 环境变量
dotenv.config();

// 初始化兼容OpenAI接口的大模型（阿里云通义系列）
const model = new ChatOpenAI({
  modelName: process.env.MODEL_NAME,
  apiKey: process.env.OPENAI_API_KEY,
  configuration: {
    baseURL: process.env.OPENAI_BASE_URL,
  },
});

// 发起请求并打印结果
const response = await model.invoke("介绍下自己");
console.log(response.content);