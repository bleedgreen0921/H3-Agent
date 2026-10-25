import fs from 'node:fs';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const { AllTools } = require('@larksuiteoapi/lark-mcp/dist/mcp-tool/tools/index.js');
const allowed = {
 im: ['im.v1.chat.get','im.v1.chat.list','im.v1.chat.search',...['get','list','create','reply','forward','update','patch'].map(x=>'im.v1.message.'+x)],
 docx: ['docx.v1.document.get','docx.v1.document.rawContent','docx.v1.document.create','docx.v1.documentBlock.get','docx.v1.documentBlock.list','docx.v1.documentBlock.patch','docx.v1.documentBlock.batchUpdate','docx.v1.documentBlockChildren.create','docx.v1.documentBlockChildren.get'],
 drive: ['drive.v1.file.list','drive.v1.file.createFolder','drive.v1.file.move'],
 wiki: ['wiki.v2.space.list','wiki.v2.space.get','wiki.v2.space.getNode','wiki.v2.spaceNode.list','wiki.v2.spaceNode.create','wiki.v2.spaceNode.move','wiki.v2.spaceNode.updateTitle'],
 bitable: ['bitable.v1.app.get','bitable.v1.appTable.list','bitable.v1.appTable.create','bitable.v1.appTableField.list','bitable.v1.appTableField.create',...['get','list','search','create','update','batchCreate','batchUpdate','batchGet'].map(x=>'bitable.v1.appTableRecord.'+x)],
 calendar: ['calendar.v4.calendar.list','calendar.v4.calendar.get',...['get','list','search','create','patch','delete'].map(x=>'calendar.v4.calendarEvent.'+x),...['list','create','batchDelete'].map(x=>'calendar.v4.calendarEventAttendee.'+x)],
 task: ['task.v2.task.get','task.v2.task.list','task.v2.task.create','task.v2.task.patch','task.v2.task.addMembers','task.v2.task.removeMembers','task.v2.task.addTasklist','task.v2.task.tasklists','task.v2.tasklist.get','task.v2.tasklist.list','task.v2.tasklist.create','task.v2.tasklist.patch','task.v2.tasklist.tasks'],
 contact: ['contact.v3.user.get','contact.v3.user.batchGetId'],
};
const scopes = {im:['im:message','im:message:readonly','im:chat:readonly'],docx:['docx:document'],drive:['drive:drive'],wiki:['wiki:wiki'],bitable:['bitable:app'],calendar:['calendar:calendar'],task:['task:task'],contact:['contact:user.base:readonly']};
const read = new Set(['get','getNode','list','search','rawContent','batchGet','batchGetId','tasklists','tasks']);
const tools = [];
for (const [domain,names] of Object.entries(allowed)) for (const name of names) {
 const tool=AllTools.find(t=>t.name===name); if(!tool) throw Error('Missing official tool '+name);
 tools.push({name, native_name:name.replaceAll('.','_'), domain, write:!read.has(name.split('.').at(-1)),
 identities:tool.accessTokens.map(x=>x==='tenant'?'bot':'user'), scopes:scopes[domain], path:tool.path, method:tool.httpMethod});
}
const target=new URL('../../src/ai_assistant/integrations/feishu-capabilities.json',import.meta.url);
fs.writeFileSync(target,JSON.stringify({version:1,official_mcp_version:'0.5.1',tools},null,2)+'\n');
