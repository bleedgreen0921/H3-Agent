// No SDK or account access: trusted validation and durable admission.
import crypto from 'node:crypto';
// 仅补充私聊 PDF、DOCX、Markdown 附件 post：正文与附件引用作为同一次请求入队。
function attachmentPostText(content,messageId){
 if(!content||typeof content!=='object'||!Array.isArray(content.files))return null;
 const files=content.files.filter(file=>file&&file.is_folder!==true&&
  typeof file.file_key==='string'&&file.file_key.length>0&&
  typeof file.file_name==='string'&&/\.(pdf|docx|md)$/i.test(file.file_name));
 if(!files.length)return null;
 const parts=[];
 if(typeof content.title==='string'&&content.title.trim())parts.push(content.title);
 // content_v2 是同一正文的另一表示，不重复拼接。
 const rows=Array.isArray(content.content)?content.content:content.content_v2;
 if(Array.isArray(rows))for(const row of rows){
  if(!Array.isArray(row))continue;
  const text=row.filter(node=>node?.tag==='text'&&typeof node.text==='string').map(node=>node.text).join('');
  if(text)parts.push(text);
 }
 for(const file of files)parts.push(`File reference: ${JSON.stringify({file_key:file.file_key,file_name:file.file_name,message_id:messageId})}`);
 return parts.join('\n');
}
export function validateEvent(config,botId,event){
 const m=event.message,sender=event.sender;
 if(sender?.sender_type!=='user'||sender.sender_id?.open_id!==config.owner_open_id)return null;
 if(!m||!['text','file','post'].includes(m.message_type)||typeof m.message_id!=='string'||typeof m.chat_id!=='string')return null;
 const group=m.chat_type==='group';
 if(group&&m.message_type==='post')return null;
 if(group&&(!config.groups[m.chat_id]||!m.mentions?.some(x=>x.id?.open_id===botId)))return null;
 if(!group&&m.chat_type!=='p2p')return null;
 let content;try{content=JSON.parse(m.content);}catch{return null;}
 if(!content||typeof content!=='object')return null;
 let text=m.message_type==='post'?attachmentPostText(content,m.message_id):m.message_type==='text'?content.text:`File reference: ${JSON.stringify({file_key:content.file_key,file_name:content.file_name,message_id:m.message_id})}`;
 if(typeof text!=='string'||text.length>100000)return null;
 for(const mention of m.mentions||[])text=text.replaceAll(mention.key,'').trim();
 return {id:m.message_id,binding:`${config.app_id}:${m.chat_id}:${config.owner_open_id}`,
  body:{text,chat_id:m.chat_id,group,owner_open_id:config.owner_open_id}};
}
export async function persistInbox(db,input,config){
 try{
  await db.query('BEGIN');
  await db.query('SELECT pg_advisory_xact_lock(714160003)');
  if((await db.query('SELECT 1 FROM assistant_inbox WHERE id=$1',[input.id])).rowCount){await db.query('COMMIT');return 'duplicate';}
  const count=Number((await db.query("SELECT count(*) FROM assistant_inbox WHERE status IN ('received','running')")).rows[0].count);
  const busy=count>=100&&!['/cancel','/new'].includes(input.body.text);
  await db.query('INSERT INTO assistant_inbox(id,binding,body,status) VALUES($1,$2,$3,$4)',
   [input.id,input.binding,input.body,busy?'completed':'received']);
  const now=Date.now()/1000;
  if(busy)await db.query('INSERT INTO assistant_outbox(id,input_id,body) VALUES($1,$2,$3)',
   [crypto.randomUUID(),input.id,{chat_id:input.body.chat_id,text:'当前队列已满，请稍后重试。',message:{msg_type:'post',content:JSON.stringify({zh_cn:{content:[[{tag:'md',text:'当前队列已满，请稍后重试。'}]]}})},delivery:{version:1,attempts:0,terminal_attempts:0,first_terminal_at:now,deadline:now+config.final_delivery_timeout_seconds,last_error:null,purpose:'admission'}}]);
  await db.query('COMMIT');return busy?'busy':'received';
 }catch(error){await db.query('ROLLBACK');throw error;}
}
