// 自动读取项目根目录下的 .env 文件，并把配置放进 process.env。
import "dotenv/config";

// 从 Node.js 自带的 path 模块中导入 parse，用来拆解文件路径和文件名。
import { parse } from 'path';

// 导入操作 Milvus 所需的客户端、字段类型、相似度算法和索引类型。
import { MilvusClient, DataType, MetricType, IndexType } from '@zilliz/milvus2-sdk-node';

// 导入“文本转向量”工具，它能把文字转换成一串表达文字含义的数字。
import { OpenAIEmbeddings } from "@langchain/openai";

// 导入 EPUB 读取器，用它读取电子书并按章节生成文档对象。
import { EPubLoader } from "@langchain/community/document_loaders/fs/epub";

// 导入文本拆分器，用它把很长的章节继续切成较短的文字片段。
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

// Milvus 集合名称。集合可以通俗地理解成数据库里的一张表。
const COLLECTION_NAME = 'ebook_collection';

// 每段文字生成的向量都固定包含 1024 个数字。
const VECTOR_DIM = 1024;

// 每个文字片段的目标长度是 500 个字符。
const CHUNK_SIZE = 500;

// 要处理的 EPUB 文件路径；这里表示当前运行目录下的《天龙八部》电子书。
const EPUB_FILE = './天龙八部.epub';

// parse(EPUB_FILE) 会拆解文件路径，.name 只取不带扩展名的文件名。
// 例如“./天龙八部.epub”最后会得到“天龙八部”。
const BOOK_NAME = parse(EPUB_FILE).name;

// 创建“文本转向量”工具，并配置它使用的密钥、模型和接口地址。
const embeddings = new OpenAIEmbeddings({
  // 从 .env 中读取 API 密钥。
  apiKey: process.env.OPENAI_API_KEY,

  // 从 .env 中读取向量模型名称。
  model: process.env.EMBEDDINGS_MODEL_NAME,

  // configuration 用来存放接口地址等额外配置。
  configuration: {
    // 从 .env 中读取 API 的基础地址。
    baseURL: process.env.OPENAI_BASE_URL
  },

  // 要求模型返回 1024 维向量，必须和 Milvus 中 vector 字段的维度一致。
  dimensions: VECTOR_DIM
});

// 创建 Milvus 客户端，后面的数据库操作都通过 client 完成。
const client = new MilvusClient({
  // Milvus 服务运行在本机的 19530 端口。
  address: 'localhost:19530'
});

/**
 * 把一段普通文字转换成向量。
 * @param {string} text 要转换的文字。
 * @returns {Promise<number[]>} 最终会得到一个由数字组成的向量。
 */
async function getEmbedding(text) {
  // 调用向量模型；await 表示等接口返回结果以后再继续执行。
  const result = await embeddings.embedQuery(text);

  // 把得到的向量返回给调用这个函数的地方。
  return result;
}

/**
 * 确保电子书集合已经存在并且已经加载。
 * @param {string|number} bookId 书籍编号；当前函数保留了这个参数，但函数内部暂时没有使用它。
 */
async function ensureCollection(bookId) {
  // try 包住可能失败的数据库操作；出错时会进入下面的 catch。
  try {
    // 查询 Milvus，检查 ebook_collection 集合是否已经存在。
    const hasCollection = await client.hasCollection({
      // 指定要检查的集合名称。
      collection_name: COLLECTION_NAME
    });

    // hasCollection.value 为 false，说明集合不存在，需要创建。
    if (!hasCollection.value) {
      // 在控制台提示：准备创建集合。
      console.log('创建集合...');

      // 调用 Milvus 的创建集合接口，并等待创建完成。
      await client.createCollection({
        // 使用 ebook_collection 作为集合名称。
        collection_name: COLLECTION_NAME,

        // fields 是字段清单，相当于设计数据库的“表结构”。
        fields: [
          // id：字符串主键，最长 100 个字符；每个文字片段的 id 必须唯一。
          { name: 'id', data_type: DataType.VarChar, max_length: 100, is_primary_key: true },

          // book_id：字符串，最长 100 个字符；用来区分不同的书。
          { name: 'book_id', data_type: DataType.VarChar, max_length: 100 },

          // book_name：字符串，最长 200 个字符；保存电子书名称。
          { name: 'book_name', data_type: DataType.VarChar, max_length: 200 },

          // chapter_num：32 位整数；保存当前片段属于第几章。
          { name: 'chapter_num', data_type: DataType.Int32 },

          // index：32 位整数；保存当前片段在本章中的顺序编号。
          { name: 'index', data_type: DataType.Int32 },

          // content：字符串，最长 10000 个字符；保存拆分后的正文。
          { name: 'content', data_type: DataType.VarChar, max_length: 10000 },

          // vector：1024 维浮点数向量；保存正文转换出来的数字特征。
          { name: 'vector', data_type: DataType.FloatVector, dim: VECTOR_DIM }
        ]
      });

      // 提示集合创建成功。
      console.log('✓ 集合创建成功');

      // 提示：准备为 vector 字段创建向量索引。
      console.log('创建索引...');

      // 调用创建索引接口，并等待创建完成。
      await client.createIndex({
        // 指定索引属于哪个集合。
        collection_name: COLLECTION_NAME,

        // 指定给 vector 字段创建索引。
        field_name: 'vector',

        // IVF_FLAT 会先给向量分组，搜索时优先检查相关的小组，从而减少比较次数。
        index_type: IndexType.IVF_FLAT,

        // 用余弦相似度判断两段文字的意思是否接近。
        metric_type: MetricType.COSINE,

        // 创建索引时，把向量空间计划划分成 1024 个小组。
        params: { nlist: 1024 }
      });

      // 提示索引创建成功。
      console.log('✓ 索引创建成功');
    }

    // 无论集合是刚创建的还是原来就有，都尝试把它加载到内存中。
    try {
      // 等待 Milvus 加载集合；加载后才能方便地进行向量搜索。
      await client.loadCollection({ collection_name: COLLECTION_NAME });

      // 没有抛出错误，表示这次加载成功。
      console.log('✓ 集合已加载');
    // 如果加载时抛出错误，就进入这里。
    } catch (error) {
      // 原代码把加载错误当成“集合已经加载”来处理，所以这里只打印提示，不再抛出错误。
      console.log('✓ 集合已处于加载状态');
    }
  // 如果检查、创建集合或创建索引失败，就进入这里。
  } catch (error) {
    // 输出简短的错误原因。
    console.error('创建集合时出错:', error.message);

    // 把错误继续抛给上层函数，让整个程序知道这里失败了。
    throw error;
  }
}

/**
 * 为一个章节中的所有文字片段生成向量，然后整批插入 Milvus。
 * @param {string[]} chunks 本章拆出来的文字片段数组。
 * @param {string|number} bookId 书籍编号。
 * @param {number} chapterNum 章节编号。
 * @returns {Promise<number>} 最终会返回实际插入的记录数量。
 */
async function insertChunksBatch(chunks, bookId, chapterNum) {
  // 捕获生成向量或写数据库时出现的错误。
  try {
    // 如果这一章没有任何文字片段，就不调用接口，直接返回插入了 0 条。
    if (chunks.length === 0) {
      return 0;
    }

    // 处理本章的全部片段，并等待所有片段都生成向量。
    const insertData = await Promise.all(
      // map 会逐个处理 chunks；chunk 是正文，chunkIndex 是它在数组中的编号。
      chunks.map(async (chunk, chunkIndex) => {
        // 把当前文字片段转换成向量。
        const vector = await getEmbedding(chunk);

        // 返回一条可以写进 Milvus 的完整记录。
        return {
          // 用“书籍编号_章节编号_片段编号”拼出唯一 id。
          id: `${bookId}_${chapterNum}_${chunkIndex}`,

          // 保存这条记录属于哪一本书。
          book_id: bookId,

          // 保存从 EPUB 文件名中提取出来的书名。
          book_name: BOOK_NAME,

          // 保存当前章节编号。
          chapter_num: chapterNum,

          // 保存片段在当前章节中的位置编号。
          index: chunkIndex,

          // 保存文字片段本身。
          content: chunk,

          // 保存刚生成的向量；这里也可以简写成只写 vector。
          vector: vector
        };
      })
    );

    // 把当前章节已经带有向量的所有片段一次性插入 Milvus。
    const insertResult = await client.insert({
      // 指定数据要写入 ebook_collection 集合。
      collection_name: COLLECTION_NAME,

      // insertData 是本章全部待插入记录组成的数组。
      data: insertData
    });

    // insert_cnt 可能不是普通数字，所以先用 Number 转换；转换失败或没有值时返回 0。
    return Number(insertResult.insert_cnt) || 0;
  // 如果生成向量或插入数据失败，就进入这里。
  } catch (error) {
    // 输出发生错误的章节编号和简短原因。
    console.error(`插入章节 ${chapterNum} 的数据时出错:`, error.message);

    // 再输出完整错误对象，方便排查更详细的问题。
    console.error('错误详情:', error);

    // 把错误继续抛给上层函数，停止后续处理。
    throw error;
  }
}

/**
 * 加载 EPUB，按章节处理；每处理完一章，就立刻把这一章写入 Milvus。
 * @param {string|number} bookId 书籍编号。
 * @returns {Promise<number>} 最终会返回全书成功插入的记录总数。
 */
async function loadAndProcessEPubStreaming(bookId) {
  // 捕获读取、拆分或插入 EPUB 时发生的错误。
  try {
    // 打印正在读取的电子书路径；开头的 \n 用来先空一行。
    console.log(`\n开始加载 EPUB 文件: ${EPUB_FILE}`);

    // 创建 EPUB 读取器，并告诉它要读取哪个文件、怎样拆分。
    const loader = new EPubLoader(
      // 第一个参数是 EPUB 文件路径。
      EPUB_FILE,

      // 第二个参数是读取 EPUB 时的配置。
      {
        // true 表示按照电子书章节拆成多个文档对象。
        splitChapters: true,
      }
    );

    // 真正读取 EPUB 文件，并等待读取完成。
    const documents = await loader.load();

    // documents.length 是读取到的章节数量。
    console.log(`✓ 加载完成，共 ${documents.length} 个章节\n`);

    // 创建文字拆分器，把每一个较长的章节继续拆成小片段。
    const textSplitter = new RecursiveCharacterTextSplitter({
      // 每个片段的目标长度为 500 个字符。
      chunkSize: CHUNK_SIZE,

      // 相邻片段重复保留 50 个字符，避免切割位置前后的上下文完全断开。
      chunkOverlap: 50,
    });

    // 记录到目前为止一共成功插入了多少个片段，初始值为 0。
    let totalInserted = 0;

    // 从第一章开始循环，一直处理到最后一章。
    for (let chapterIndex = 0; chapterIndex < documents.length; chapterIndex++) {
      // 通过当前下标取出这一章的文档对象。
      const chapter = documents[chapterIndex];

      // 从章节对象中取出纯正文。
      const chapterContent = chapter.pageContent;

      // chapterIndex 从 0 开始，所以显示给人看时加 1；同时显示全书总章节数。
      console.log(`处理第 ${chapterIndex + 1}/${documents.length} 章...`);

      // 把当前章节正文进一步拆成大约 500 字一段的字符串数组。
      const chunks = await textSplitter.splitText(chapterContent);

      // 显示这一章最终拆出了多少个片段。
      console.log(`  拆分为 ${chunks.length} 个片段`);

      // 如果这一章拆分后没有内容，就不生成向量，也不写数据库。
      if (chunks.length === 0) {
        // 提示跳过空章节；\n 表示输出后再空一行。
        console.log(`  跳过空章节\n`);

        // continue 表示立即结束本轮循环，直接开始处理下一章。
        continue;
      }

      // 提示：开始处理这一章的全部文字片段。
      console.log(`  生成向量并插入中...`);

      // 为本章所有片段生成向量并整批插入；显示的章节号从 1 开始，所以这里加 1。
      const insertedCount = await insertChunksBatch(chunks, bookId, chapterIndex + 1);

      // 把本章插入数量累加到全书总数中；等价于 totalInserted = totalInserted + insertedCount。
      totalInserted += insertedCount;

      // 显示本章插入数量和当前累计数量。
      console.log(`  ✓ 已插入 ${insertedCount} 条记录（累计: ${totalInserted}）\n`);
    }

    // 所有章节处理完成后，显示全书一共插入了多少条记录。
    console.log(`\n总共插入 ${totalInserted} 条记录\n`);

    // 把总数返回给调用这个函数的地方。
    return totalInserted;
  // 如果读取 EPUB、拆分文字或插入数据失败，就进入这里。
  } catch (error) {
    // 输出简短的错误原因。
    console.error('加载 EPUB 文件时出错:', error.message);

    // 把错误继续交给 main 函数处理。
    throw error;
  }
}

/**
 * 主函数：按照顺序组织整个电子书入库流程。
 */
async function main() {
  // 捕获整个程序运行期间没有被处理掉的错误。
  try {
    // '='.repeat(80) 会生成 80 个等号，用作控制台分隔线。
    console.log('='.repeat(80));

    // 打印程序标题。
    console.log('电子书处理程序');

    // 再打印一条由 80 个等号组成的分隔线。
    console.log('='.repeat(80));

    // 提示：准备连接 Milvus；开头的 \n 表示先空一行。
    console.log('\n连接 Milvus...');

    // 等待 Milvus 客户端连接完成。
    await client.connectPromise;

    // 提示连接成功。
    console.log('✓ 已连接\n');

    // 为当前电子书设置编号 1；以后处理多本书时，可以给每本书不同的编号。
    const bookId = 1;

    // 检查集合是否存在；没有就创建，最后确保它已经加载。
    await ensureCollection(bookId);

    // 读取 EPUB，按章拆分、生成向量，并一章一章写入 Milvus。
    await loadAndProcessEPubStreaming(bookId);

    // 打印结束分隔线。
    console.log('='.repeat(80));

    // 提示整本书已经处理完成。
    console.log('处理完成！');

    // 再打印一条结束分隔线。
    console.log('='.repeat(80));
  // main 中任意未处理的错误最终都会进入这里。
  } catch (error) {
    // 输出简短错误原因。
    console.error('\n错误:', error.message);

    // 输出完整调用栈，帮助定位错误发生在哪个函数、哪一行。
    console.error(error.stack);

    // 用状态码 1 结束 Node.js 进程，表示程序运行失败。
    process.exit(1);
  }
}

// 调用主函数，正式启动电子书处理程序。
main();
