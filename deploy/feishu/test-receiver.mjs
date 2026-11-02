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

assert.deepEqual(saved.message,{msg_type:'post',content:JSON.stringify({zh_cn:{content:[[{tag:'md',text:saved.text}]]}})});

// 复现实际 PDF 消息结构：post.files 与正文同消息，content_v2 不重复。
const pdfContent={title:'PDF 分析',content:[[{tag:'text',text:'分析这份 PDF，给出介绍',style:[]}]],
 content_v2:[[{tag:'text',text:'分析这份 PDF，给出介绍',style:[]}]],
 files:[{file_key:'pdf-key',file_name:'fixture.pdf',is_folder:false}]};
const pdfEvent={...event,message:{...event.message,chat_type:'p2p',chat_id:'private',
 message_type:'post',mentions:[],content:JSON.stringify(pdfContent)}};
const pdfInput=validateEvent(config,'bot',pdfEvent);
assert.equal(pdfInput.body.group,false);
assert.equal(pdfInput.body.text,'PDF 分析\n分析这份 PDF，给出介绍\nFile reference: '+JSON.stringify({file_key:'pdf-key',file_name:'fixture.pdf',message_id:'message'}));
assert.equal(validateEvent({...config,owner_open_id:'other'},'bot',pdfEvent),null);
assert.equal(validateEvent(config,'bot',{...pdfEvent,sender:{...pdfEvent.sender,sender_type:'app'}}),null);
assert.equal(validateEvent(config,'bot',{...pdfEvent,message:{...pdfEvent.message,chat_type:'group',chat_id:'group',mentions:event.message.mentions}}),null);
const postWith=content=>validateEvent(config,'bot',{...pdfEvent,message:{...pdfEvent.message,content:JSON.stringify(content)}});
for(const content of [null,{}, {content:pdfContent.content}, {...pdfContent,files:{}},
 {...pdfContent,files:[null,{}, {file_key:'key',file_name:'folder.pdf',is_folder:true}]},
 {...pdfContent,files:[{file_key:'key',file_name:'fixture.docx'}]},
 {...pdfContent,files:[{file_key:'',file_name:'fixture.pdf'}]}])assert.equal(postWith(content),null);
assert.equal(validateEvent(config,'bot',{...pdfEvent,message:{...pdfEvent.message,content:'{invalid'}}),null);
assert.equal(postWith({...pdfContent,title:'x'.repeat(100001)}),null);
const multiple=postWith({content_v2:pdfContent.content_v2,files:[
 {file_key:'first',file_name:'FIRST.PDF'}, {file_key:'second',file_name:'second.pdf'}]});
assert.match(multiple.body.text,/分析这份 PDF/);
assert.equal(multiple.body.text.split('File reference:').length-1,2);
const attachmentOnly=postWith({files:pdfContent.files});
assert.match(attachmentOnly.body.text,/^File reference:/);
const ordinaryFile=validateEvent(config,'bot',{...pdfEvent,message:{...pdfEvent.message,
 message_type:'file',content:JSON.stringify(pdfContent.files[0])}});
assert.equal(ordinaryFile.body.text,attachmentOnly.body.text);
// 与普通消息使用相同事务及 message_id 去重，不自动重放历史漏收消息。
let inserted;
let duplicate=false;
const pdfDb={query:async(sql,args)=>{
 if(sql.startsWith('SELECT 1'))return {rowCount:duplicate?1:0,rows:[]};
 if(sql.startsWith('SELECT count'))return {rows:[{count:0}]};
 if(sql.startsWith('INSERT INTO assistant_inbox')){inserted=args;duplicate=true;}
 return {rows:[],rowCount:1};
}};
assert.equal(await persistInbox(pdfDb,pdfInput,config),'received');
assert.deepEqual(inserted,['message','app:private:owner',pdfInput.body,'received']);
assert.equal(await persistInbox(pdfDb,pdfInput,config),'duplicate');
console.log('Private PDF post text, attachment references, admission/deduplication and existing boundaries passed.');
