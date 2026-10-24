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
