/*
 * @Author: dali crayoncrook@gmail.com
 * @Date: 2026-08-28 11:08:12
 * @LastEditors: dali crayoncrook@gmail.com
 * @LastEditTime: 2026-08-28 11:29:52
 * @FilePath: \tool-test\src\delete.mjs
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
import "dotenv/config";
import { MilvusClient } from '@zilliz/milvus2-sdk-node';

const COLLECTION_NAME = 'ai_diary';

const client = new MilvusClient({
  address: 'localhost:19530'
});

async function main() {
  try {
    console.log('Connecting to Milvus...');
    await client.connectPromise;
    console.log('✓ Connected\n');

    // 删除单条数据
    console.log('Deleting diary entry...');
    const deleteId = 'diary_005';

    const result = await client.delete({
      collection_name: COLLECTION_NAME,
      filter: `id == "${deleteId}"`
    });

    console.log(`✓ Deleted ${result.delete_cnt} record(s)`);
    console.log(`  ID: ${deleteId}\n`);

    // 批量删除
    console.log('Batch deleting diary entries...');
    const deleteIds = ['diary_002', 'diary_003'];
    const idsStr = deleteIds.map(id => `"${id}"`).join(', ');

    const batchResult = await client.delete({
      collection_name: COLLECTION_NAME,
      filter: `id in [${idsStr}]`
    });

    console.log(`✓ Batch deleted ${batchResult.delete_cnt} record(s)`);
    console.log(`  IDs: ${deleteIds.join(', ')}\n`);

    // 条件删除
    console.log('Deleting by condition...');
    const conditionResult = await client.delete({
      collection_name: COLLECTION_NAME,
      filter: `mood == "sad"`
    });

    console.log(`✓ Deleted ${conditionResult.delete_cnt} record(s) with mood="sad"\n`);

  } catch (error) {
    console.error('Error:', error.message);
  }
}

main();
