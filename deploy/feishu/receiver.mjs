// Trusted receiver: SDK callbacks only validate and commit an inbox event.
import * as lark from '@larksuiteoapi/node-sdk';
import pg from 'pg';
import readline from 'node:readline';
import {validateEvent,persistInbox} from './receiver-core.mjs';
const lines = readline.createInterface({input: process.stdin});
const config = JSON.parse(await new Promise(resolve=>lines.once('line', resolve)));
lines.close();
const pool = new pg.Pool({connectionString: config.database_url, max: 2, connectionTimeoutMillis: 10000});
const client = new lark.Client({appId: config.app_id, appSecret: config.app_secret,
  appType: lark.AppType.SelfBuild, loggerLevel: lark.LoggerLevel.error});
const botInfo = await client.request({url:'/open-apis/bot/v3/info',method:'GET'});
const botId = botInfo.bot?.open_id;
if (!botId) throw Error('Unable to resolve bot identity');
const dispatcher = new lark.EventDispatcher({}).register({
 'im.message.receive_v1': async event => {
  const input=validateEvent(config,botId,event);
  if(!input)return;
  const db=await pool.connect();
  try{await persistInbox(db,input);}finally{db.release();}
 }
});
const socket = new lark.WSClient({appId:config.app_id,appSecret:config.app_secret,loggerLevel:lark.LoggerLevel.error});
await socket.start({eventDispatcher:dispatcher});
for(const signal of ['SIGTERM','SIGINT']) process.on(signal,async()=>{socket.close();await pool.end();process.exit(0);});
process.stdin.resume();
