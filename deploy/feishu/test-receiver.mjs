import assert from 'node:assert/strict';
import {validateEvent,persistInbox} from './receiver-core.mjs';
const config={app_id:'app',final_delivery_timeout_seconds:90,owner_open_id:'owner',groups:{group:true}};
const event={sender:{sender_type:'user',sender_id:{open_id:'owner'}},message:{message_id:'message',chat_id:'group',chat_type:'group',message_type:'text',content:JSON.stringify({text:'@bot hello'}),mentions:[{key:'@bot',id:{open_id:'bot'}}]}};
assert.equal(validateEvent(config,'bot',event).body.text,'hello');
assert.equal(validateEvent(config,'other',event),null);
assert.equal(validateEvent({...config,owner_open_id:'other'},'bot',event),null);
assert.equal(validateEvent({...config,groups:{}},'bot',event),null);
assert.equal(validateEvent(config,'bot',{...event,sender:{...event.sender,sender_type:'app'}}),null);
const queries=[];
await assert.rejects(persistInbox({query:async sql=>{queries.push(sql);if(sql.startsWith('SELECT 1'))throw Error('DB unavailable');return {rows:[]};}},validateEvent(config,'bot',event)));
assert.equal(queries.at(-1),'ROLLBACK');
console.log('Receiver owner, bot mention, group grants and persistence failure checks passed.');
// 队列已满时，退役的 /status 不再绕过容量限制。
for(const [text,expected] of [['/status','busy'],['/cancel','received'],['/new','received']]){
 const db={query:async(sql)=>sql.startsWith('SELECT 1')?{rowCount:0,rows:[]}:sql.startsWith('SELECT count')?{rows:[{count:100}]}:{rowCount:1,rows:[]}};
 const input=validateEvent(config,'bot',{...event,message:{...event.message,content:JSON.stringify({text:'@bot '+text})}});
 assert.equal(await persistInbox(db,input,config),expected);
}

// 队满回复必须使用当前元数据，并与 inbox 在同一事务登记。
let saved;
const busyQueries=[];
const busyDb={query:async(sql,args)=>{
 busyQueries.push(sql);
 if(sql.startsWith('SELECT 1'))return {rowCount:0,rows:[]};
 if(sql.startsWith('SELECT count'))return {rows:[{count:100}]};
 if(sql.startsWith('INSERT INTO assistant_outbox'))saved=args[2];
 return {rowCount:1,rows:[]};
}};
assert.equal(await persistInbox(busyDb,validateEvent(config,'bot',event),config),'busy');
assert.equal(saved.delivery.version,1);
assert.equal(saved.delivery.deadline-saved.delivery.first_terminal_at,90);
assert.equal(saved.delivery.terminal_attempts,0);
assert.equal(busyQueries[0],'BEGIN');
assert.equal(busyQueries.at(-1),'COMMIT');
console.log('Busy admission uses current delivery metadata and one transaction.');
