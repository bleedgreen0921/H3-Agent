// No SDK or account access: trusted validation and durable admission.
import crypto from 'node:crypto';
export function validateEvent(config,botId,event){
 const m=event.message,sender=event.sender;
 if(sender?.sender_type!=='user'||sender.sender_id?.open_id!==config.owner_open_id)return null;
 if(!m||!['text','file'].includes(m.message_type)||typeof m.message_id!=='string'||typeof m.chat_id!=='string')return null;
 const group=m.chat_type==='group';
 if(group&&(!config.groups[m.chat_id]||!m.mentions?.some(x=>x.id?.open_id===botId)))return null;
 if(!group&&m.chat_type!=='p2p')return null;
 let content;try{content=JSON.parse(m.content);}catch{return null;}
 let text=m.message_type==='text'?content.text:`File reference: ${JSON.stringify({file_key:content.file_key,file_name:content.file_name,message_id:m.message_id})}`;
 if(typeof text!=='string'||text.length>100000)return null;
 for(const mention of m.mentions||[])text=text.replaceAll(mention.key,'').trim();
 return {id:m.message_id,binding:`${config.app_id}:${m.chat_id}:${config.owner_open_id}`,
  body:{text,chat_id:m.chat_id,group,owner_open_id:config.owner_open_id}};
}
export async function persistInbox(db,input){
 try{
  await db.query('BEGIN');
  await db.query('SELECT pg_advisory_xact_lock(714160003)');
  if((await db.query('SELECT 1 FROM assistant_inbox WHERE id=$1',[input.id])).rowCount){await db.query('COMMIT');return 'duplicate';}
  const count=Number((await db.query("SELECT count(*) FROM assistant_inbox WHERE status IN ('received','running','waiting')")).rows[0].count);
  const busy=count>=100&&!['/cancel','/status','/new'].includes(input.body.text);
  await db.query('INSERT INTO assistant_inbox(id,binding,body,status) VALUES($1,$2,$3,$4)',
   [input.id,input.binding,input.body,busy?'completed':'received']);
  if(busy)await db.query('INSERT INTO assistant_outbox(id,input_id,body) VALUES($1,$2,$3)',
   [crypto.randomUUID(),input.id,{chat_id:input.body.chat_id,text:'当前队列已满，请稍后重试。'}]);
  await db.query('COMMIT');return busy?'busy':'received';
 }catch(error){await db.query('ROLLBACK');throw error;}
}
