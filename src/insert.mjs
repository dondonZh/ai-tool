// 自动读取项目根目录中的 .env 文件，并把里面的配置放进 process.env。
// 例如 .env 里的 OPENAI_API_KEY，可以通过 process.env.OPENAI_API_KEY 取到。
import "dotenv/config";

// 导入操作 Milvus 需要的工具：客户端、字段类型、相似度算法和索引类型。
import { MilvusClient, DataType, MetricType, IndexType } from '@zilliz/milvus2-sdk-node';

// 导入“文本转向量”工具。向量就是一串用来表达文字含义的数字。
import { OpenAIEmbeddings } from "@langchain/openai";

// 给要创建的 Milvus 集合起名。集合可以通俗地理解成数据库里的一张表。
const COLLECTION_NAME = 'ai_diary';

// 每个文本向量固定包含 1024 个数字；建表和生成向量时必须使用相同的维度。
const VECTOR_DIM = 1024;

// 创建一个“文本转向量”的工具，并告诉它该使用哪个接口和模型。
const embeddings = new OpenAIEmbeddings({
  // 从环境变量中读取 API 密钥，用它证明我们有权调用向量模型。
  apiKey: process.env.OPENAI_API_KEY,

  // 从环境变量中读取向量模型的名称。
  model: process.env.EMBEDDINGS_MODEL_NAME,

  // configuration 用来放接口地址之类的额外配置。
  configuration: {
    // 读取 API 的基础地址，方便使用官方接口或兼容 OpenAI 的其他接口。
    baseURL: process.env.OPENAI_BASE_URL
  },

  // 要求模型返回 1024 维向量，必须和 VECTOR_DIM 以及 Milvus 字段维度一致。
  dimensions: VECTOR_DIM
});

// 创建 Milvus 客户端，后面通过 client 操作向量数据库。
const client = new MilvusClient({
  // Milvus 运行在本机，服务端口是 19530。
  address: 'localhost:19530'
});

// 定义异步函数：传入普通文字，返回这段文字对应的向量。
async function getEmbedding(text) {
  // 调用向量模型处理 text；await 表示等结果回来后再继续执行。
  const result = await embeddings.embedQuery(text);

  // 把得到的向量交还给调用这个函数的地方。
  return result;
}

// main 是整个脚本的主函数，所有主要操作都从这里开始。
async function main() {
  // try 包住可能失败的操作；出错时会跳到下面的 catch。
  try {
    // 在控制台提示：准备连接 Milvus。
    console.log('Connecting to Milvus...');

    // 等待 Milvus 客户端连接完成。
    await client.connectPromise;

    // 提示连接成功；\n 表示再多换一行，让输出更易读。
    console.log('✓ Connected\n');

    // 创建集合。集合类似关系型数据库中的“表”。
    console.log('Creating collection...');

    // 调用 Milvus 的创建集合接口，并等待创建完成。
    await client.createCollection({
      // 使用前面定义的 ai_diary 作为集合名称。
      collection_name: COLLECTION_NAME,

      // fields 是字段清单，相当于在这里设计“表结构”。
      fields: [
        // id：字符串，最长 50 个字符，并且是主键；每条数据的 id 必须唯一。
        { name: 'id', data_type: DataType.VarChar, max_length: 50, is_primary_key: true },

        // vector：浮点数向量，固定 1024 维；保存日记正文转换出来的数字特征。
        { name: 'vector', data_type: DataType.FloatVector, dim: VECTOR_DIM },

        // content：字符串，最长 5000 个字符；保存日记正文。
        { name: 'content', data_type: DataType.VarChar, max_length: 5000 },

        // date：字符串，最长 50 个字符；保存写日记的日期。
        { name: 'date', data_type: DataType.VarChar, max_length: 50 },

        // mood：字符串，最长 50 个字符；保存当时的心情，例如 happy。
        { name: 'mood', data_type: DataType.VarChar, max_length: 50 },

        // tags：字符串数组，最多放 10 个标签，每个标签最长 50 个字符。
        { name: 'tags', data_type: DataType.Array, element_type: DataType.VarChar, max_capacity: 10, max_length: 50 }
      ]
    });

    // 提示集合已经创建完成。
    console.log('Collection created');

    // 准备为 vector 字段创建索引。索引就像帮助数据库快速查找的“目录”。
    console.log('\nCreating index...');

    // 调用 Milvus 的创建索引接口，并等待创建完成。
    await client.createIndex({
      // 指定在哪个集合上创建索引。
      collection_name: COLLECTION_NAME,

      // 指定要给 vector 字段创建索引。
      field_name: 'vector',

      // IVF_FLAT 会先把向量分组，搜索时优先去相关的小组里查，减少比较次数。
      index_type: IndexType.IVF_FLAT,

      // 用余弦相似度比较两个向量；越接近，通常表示两段文字的意思越相似。
      metric_type: MetricType.COSINE,

      // 建索引时，计划把向量空间划分成 1024 个小组。
      params: { nlist: 1024 }
    });

    // 提示索引已经创建完成。
    console.log('Index created');

    // 准备把集合加载到内存，让它可以进行向量搜索。
    console.log('\nLoading collection...');

    // 调用加载集合接口，并等待加载完成。
    await client.loadCollection({ collection_name: COLLECTION_NAME });

    // 提示集合已经加载完成。
    console.log('Collection loaded');

    // 提示：准备插入日记数据。
    console.log('\nInserting diary entries...');

    // 这是一个数组，里面暂时放 5 条“还没有向量”的原始日记。
    const diaryContents = [
      // 第一条日记对象。
      {
        // 这条日记的唯一编号。
        id: 'diary_001',
        // 这条日记的正文。
        content: '今天天气很好，去公园散步了，心情愉快。看到了很多花开了，春天真美好。',
        // 写日记的日期。
        date: '2026-01-10',
        // 写日记时的心情；happy 是“开心”。
        mood: 'happy',
        // 用数组保存这条日记的多个标签。
        tags: ['生活', '散步']
      },

      // 第二条日记对象，各字段含义和第一条相同。
      {
        id: 'diary_002',
        content: '今天工作很忙，完成了一个重要的项目里程碑。团队合作很愉快，感觉很有成就感。',
        date: '2026-01-11',
        // excited 是“兴奋、振奋”。
        mood: 'excited',
        tags: ['工作', '成就']
      },

      // 第三条日记对象。
      {
        id: 'diary_003',
        content: '周末和朋友去爬山，天气很好，心情也很放松。享受大自然的感觉真好。',
        date: '2026-01-12',
        // relaxed 是“放松”。
        mood: 'relaxed',
        tags: ['户外', '朋友']
      },

      // 第四条日记对象。
      {
        id: 'diary_004',
        content: '今天学习了 Milvus 向量数据库，感觉很有意思。向量搜索技术真的很强大。',
        date: '2026-01-12',
        // curious 是“好奇”。
        mood: 'curious',
        tags: ['学习', '技术']
      },

      // 第五条日记对象。
      {
        id: 'diary_005',
        content: '晚上做了一顿丰盛的晚餐，尝试了新菜谱。家人都说很好吃，很有成就感。',
        date: '2026-01-13',
        // proud 是“自豪、有成就感”。
        mood: 'proud',
        tags: ['美食', '家庭']
      }
    ];

    // 提示：现在开始把每篇日记的正文转换成向量。
    console.log('Generating embeddings...');

    // diaryData 将保存“原始日记字段 + 新生成的 vector 字段”。
    // Promise.all 会同时等待所有日记的向量都生成完，再返回完整结果。
    const diaryData = await Promise.all(
      // map 会处理数组中的每条 diary，并为每条日记生成一个新对象。
      diaryContents.map(async (diary) => ({
        // ...diary 会把原有的 id、content、date、mood、tags 全部复制过来。
        ...diary,

        // 把正文交给 getEmbedding，等模型返回向量后，将它保存到 vector 字段。
        vector: await getEmbedding(diary.content)
      }))
    );

    // 把已经带有 vector 的 5 条日记一次性插入 Milvus。
    const insertResult = await client.insert({
      // 指定数据要插入 ai_diary 集合。
      collection_name: COLLECTION_NAME,

      // diaryData 就是最终要插入的数据数组。
      data: diaryData
    });

    // 显示实际插入了多少条；${...} 会把 insert_cnt 的值放进字符串中。
    console.log(`✓ Inserted ${insertResult.insert_cnt} records\n`);
  // try 中任何一步出错，程序都会进入这里，错误对象会放进 error。
  } catch (error) {
    // 输出简洁的错误原因。
    console.error('Error:', error.message);
  }
}

// 调用 main，正式启动整个脚本。
main();
