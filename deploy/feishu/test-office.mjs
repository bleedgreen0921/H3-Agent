// Official generated handlers exercised entirely against an injected SDK fixture.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createRequire} from 'node:module';
const require=createRequire(import.meta.url);
const {AllTools}=require('@larksuiteoapi/lark-mcp/dist/mcp-tool/tools/index.js');
const {larkOapiHandler}=require('@larksuiteoapi/lark-mcp/dist/mcp-tool/utils/handler.js');
const {getShouldUseUAT}=require('@larksuiteoapi/lark-mcp/dist/mcp-tool/utils/get-should-use-uat.js');
const manifest=JSON.parse(fs.readFileSync(new URL('../../src/ai_assistant/integrations/feishu-capabilities.json',import.meta.url)));
for(const item of manifest.tools){
 const official=AllTools.find(t=>t.name===item.name);assert.ok(official,item.name);
 assert.equal(official.path,item.path);assert.equal(official.httpMethod,item.method);
 assert.ok(!/Acl|permission|user\.(create|delete|patch|update)/i.test(item.name));
}
for(const domain of ['im','docx','drive','wiki','bitable','calendar','task','contact']){
 for(const write of (domain==='contact'?[false]:[false,true])){
  const capability=manifest.tools.find(t=>t.domain===domain&&t.write===write);
  assert.ok(capability,`${domain} write=${write}`);
  const tool=AllTools.find(t=>t.name===capability.name);
  let count=0;
  const fixture={request:async request=>{count++;assert.equal(request.method,tool.httpMethod);assert.equal(request.url,tool.path);return {data:{items:[{id:'fixture'}],has_more:true,page_token:'next'}};}};
  const result=await larkOapiHandler(fixture,{path:{fixture:'id'},params:{page_size:1},data:{fixture:'value'},useUAT:false},{tool});
  assert.equal(count,1);assert.match(result.content[0].text,/next/);
  for(const code of [99991672,99991400]){
   const failure=await larkOapiHandler({request:async()=>{throw {response:{data:{code}}};}},{},{tool});
   assert.equal(failure.isError,true);assert.match(failure.content[0].text,new RegExp(String(code)));
  }
 }
}
assert.equal(getShouldUseUAT('tenant_access_token',true),false);
assert.equal(getShouldUseUAT('user_access_token',false),true);
console.log(`Verified ${manifest.tools.length} official capabilities, every domain handler, permission/rate errors and fixed identities.`);

// 文档写入使用官方 schema 和 SDK fixture；不读取真实 OAuth 或请求远端。
const {z}=require('zod');
const edit={elements:[{text_run:{content:'文档编辑 fixture'}}]};
const cases=[
 ['docx.v1.documentBlockChildren.create',{path:{document_id:'doc',block_id:'doc'},data:{children:[{block_type:2,text:{elements:edit.elements}}],index:-1}}],
 ['docx.v1.documentBlock.patch',{path:{document_id:'doc',block_id:'block'},data:{update_text_elements:edit}}],
 ['docx.v1.documentBlock.batchUpdate',{path:{document_id:'doc'},data:{requests:[{block_id:'block',update_text_elements:edit}]}}],
];
for(const [name,args] of cases){
 const tool=AllTools.find(t=>t.name===name);
 const parsed=z.object(tool.schema).parse({...args,useUAT:getShouldUseUAT('user_access_token',false)});
 assert.equal(parsed.useUAT,true);
 let calls=0;
 const fixture={request:async(request,options)=>{
  calls++;
  assert.equal(request.method,tool.httpMethod);
  assert.equal(request.url,tool.path);
  assert.deepEqual(request.path,args.path);
  assert.deepEqual(request.data,args.data);
  assert.ok(options); // 官方 handler 为用户身份传递 token 请求选项。
  return {data:{blocks:[{block_id:'block'}],document_revision_id:2}};
 }};
 const result=await larkOapiHandler(fixture,parsed,{tool,userAccessToken:'fixture-user-token'});
 assert.equal(calls,1);
 assert.equal(result.isError,undefined);
 assert.equal(JSON.parse(result.content[0].text).document_revision_id,2);
 for(const code of [99991672,99991400]){
  let failures=0;
  const failed=await larkOapiHandler({request:async()=>{failures++;throw {response:{data:{code}}};}},parsed,{tool,userAccessToken:'fixture-user-token'});
  assert.equal(failed.isError,true);
  assert.equal(failures,1);
  assert.match(failed.content[0].text,new RegExp(String(code)));
 }
}
console.log('Document append, patch and batch update schemas, user identity and error forwarding passed.');
