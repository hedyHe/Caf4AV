#coding:utf8

import json
import os
import traceback
import re
import csv
from collections import Counter, defaultdict
import numpy as np
import math
import networkx as nx
import community
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import random

random.seed(42)

dataset = 'cndbpedia'
# dataset = 'dbpedia'

globle_fw = open('emb_tfidf6_all_sim_pairs.json', 'w', encoding='utf-8')

class Sim_cal:
	def __init__(self, sim_type = 'cp_w_freq', wp_fw =None, idf_fw=None):		
		self.puncs = ' {}[]()-=+.;,/?\'"`~!@#$%^&*【】（），。/；？•‘“”’！　．\\<>《》—?×￥…|'
		data = {"女":["妇女","女女","女性","女","女【灵虫本来无性别，通过修炼可以选择成为男性或女性，后糖宝选择成为女孩】","女（动画版未详细解释清楚）","女（体）","雌《女》","性别：女","男→女（体）","女(17岁之前是男性)","女性（仅以声音辨别，电脑是无性别的）","女（官方一设）","女（汉字）","女{主人格","雌性（准确的说是女汉子。。。）","男；女（S.I.C.）","女性(大神暗示过，大神传确认)","男（曾于漫画100话首次被菈菈意外变成女生，此后又数次变成女生）","腐女","女(登场被扮成男性但小茂有察觉)","女子（汉语词语）","女性（雌性人类）","女（外表为女性，实际上神是没有性别的）","N女","b女","应为女","坤女","女性（汉语词语）","女（无性别，附在一村妇体内）","200%女","多数为女性","中老年女性","外观为女性","转性后为女","女孩子ww","女（姓氏）","富家女","女20","女(原男性)","女孩子","女（一开始只有纣王知道）","女，学艺时曾女扮男装为司音","女孩（年轻女性）","女(不包括动画)","少女（汉语词语）","女【灵虫本来无性别，通过修炼可以选择成为男性或女性，后糖宝选择成为女孩】","女（客观的讲是“雌性”）","开始是男性，后被师父改造为女性","原男现女","女生（汉语词语）","女（有争议）","小女孩（汉语词语）","﻿女","女1","女2","女♀","为女","女生","女（好像也没什么不对......）","灵族男或女","不明（美版为女）","每组女","男、女(性转后）","BUFF型女神","男女（汉语词语）","通常为女性（异性同体）"],"男/女":["男/女","女/♀"],"男":["男系","　男","男．","`男","男","男生","男（本体及灵魂）→不定（由容器的性别决定）","男；男（声，武神Faiz）","男出处：少年漫画《家庭教师》","在《数码宝贝合体战争》中为男","男（剧中有白向鸣人解释过）","男人（具有xy染色体的人）","男（兽人）","男（官方一季设定）","男（由原始程序而定）","男(28卷64页)","男（准确地说是公）","男士","男？","那男","男n","男♂","男子","男性","男同","男国","男1","男（公）","男性外观","均为男性","须眉男子","男性角色","通常为男","成年男子","Man/男","男（？）","男（雄）","x'b男","男性外表","性别：男","男（或无）","1948年9月28日男","男(游戏内可自由选择性别)","这么可爱一定是男孩子","男（客观地说是“雄性”）","男孩（年轻的男性人类）","男2","男0","男（召唤者不分）","男场上位置:后卫","秀吉（自称男性）","剧情中默认为男","男（官方一季设）","统统是男的","帅哥人气男","男（太子）","男广东大埔","男（已知）","男（推测）","男（公？）","男祖","男（雄性）","男性或中性","推测为男性","南男","男3"],"每男":["男c","每男"],"d男":["男同胞","男×男","男医师","每组男","n男","男足球","均为男","男孩子","d男","男/男"],"每组里信男":["每组里信男","男；男（外表）","男性阿斯泰坦","男（官方一设）","男（进之介）","男(外观决定)","男（外表）、男"],"母猫":["母（汉字）","母虎","母猫"],"母（公蚊吃素，母蚊吃荤--不一定，公的也會吸血）":["雌鸽","母（公蚊吃素，母蚊吃荤--不一定，公的也會吸血）","母","雌性（果宝特攻台词中用“她”称呼）","雌性","雌","雌♀","雌体"],"雄":["雄（中国汉字之一）","雄","雄性","已出场成员均为雄性","雄性动物","雄性(有争议)","起源为雄性","雄性大熊猫","一般为雄性","雄性机械羊","可能为雄性","100%雄性"],"雄性2":["雄性精灵","雄性2","雄（中国汉字之一）"],"无性":["无性别","无性"],"MM":["M/M","MM"],"未知（官方公布前请不要随意修改":["未知","未知（官方公布前请不要随意修改"],"伪娘":["男（或者“伪娘”）","伪娘（新兴词汇）","男（伪娘）","伪娘","冒失娘"],"至今不明":["至今不明","性别不明"],"不确定（定义）":["性别不安定的状态","不确定（定义）"],"男（有变化可能性）":["男（有变化可能性）","男（可以变成女性）"],"male":["male","♂1:♀1","凹凸man","GG"],"Female":["Female","female"],"男女皆可":["男女皆可","男、女(性转后）","菩萨身（不分男女）","菩萨身，不分男女","男女同体","女【灵虫本来无性别，通过修炼可以选择成为男性或女性，后糖宝选择成为女孩】","非男非女"],"男（男扮女装）":["男（男扮女装）","男(漫画未连载时最初原案设定为女)","男（常常被误以为是女孩）"],"男（原本为女性）":["男（原本为女性）","男〔出生时性别为女〕","男（常常被误以为是女孩）"],"母":["母","大多数是母猫","可爱的小母猫"],"雌":["雌","雌性或无性","雌性（果宝特攻台词中用“她”称呼）","大部分是雌","大概是雌性","100%雌性","雌（汉语汉字）","雌性（猜测）"],"雌雄同体":["雌雄同体","雌雄同体（生物学术语）"],"公":["公","公（汉语汉字）","雄性（他是公的哦）"],"变性":["变性","女（变性后）"],"noadded_vs":["非男","雌性（动画中为雄性）","♂♀","男（齐藤八云）、女（鞍马八云）","100%♂","男，但人间体是女性，超越性别的存在","阉","严格的来说妖怪没有性别，也可以变成女人的模样","男（原著小说里为男性，漫画里为女性）","无","不定","nan","变性女","不限","无性别，但偏男","第三性人妖","50%♂50%♀","男&女（暴走士道篇）","男（部分章节性别为女）","男孩和女孩","男（紫魅时为女）","男（番外6中变成因女神宝物成了萝莉）","女（男）","女（现实版本为男性）","未详","不明（推测为男性）","男和女","未知（零不分性别）","水母雌雄同体","不分（零不分男女，男是化身，女是附体墨夷）","女(穿越之前是男人)","女/男","男→女","大者为雌","男，电视剧后改为女","男女共体","男，可变身为女","男（人体间女）","妇","不详","男、未知（创世王）","男性和女性","♂（传说中爱神和女神的代用符号）","明明是女的又说自己是男的","不明","重生前为男，重生后为女","公熊","男（人妖）","不定（汉语词汇）","男（第三季为女【配音变成了女性】）","常态为雄性，可雌雄分体","软妹","女→男","小公主（美国伯内特夫人著儿童文学）","男（伪装成女人）","男（可捏造成女性形态）","公猫","公狗","未知（汉语词语）","男男","男女","男（领袖的挑战中为女）","男性用户为60%，女性用户为40%","无性别（通常女性装扮）","公鸡（公鸡）","nü","双性"]}
		docs = []
		for k, vs in data.items():
			docs.extend(vs)
		self.docs = docs

		self.sim_type = sim_type
		self.wp_fw = wp_fw
		self.idf_fw = idf_fw

		self.piece4word = {}   

		if 'text_emb' in sim_type :
			self.dataset = dataset
			if 'small' in sim_type:
				self.emb_file = self.dataset + '/embeddings_small.csv'
			elif 'large' in sim_type:
				self.emb_file = self.dataset +'/embeddings_large.csv'
			elif 'bert' in sim_type:
				self.emb_file = self.dataset +'/bert_embeddings.csv'
			print('file for embedding is', self.emb_file)
			self.all_embs = {}
			self.readembs(self.emb_file)


	def get_word_piece_freq_wo_freq(self, docs=None):
		#Get the frequency of the word piece level, taking into account how often each value appears, not how many times each document appears
		if docs is None:
			docs = {"v_total":325501,"男":249397,"男性用户约占61%，女性约39%":1,"北辰社":1,"女（汉字）":63644,"女":1,"个体不同":1,"雌性":78,"男（部分章节性别为女）":1,"雄":33,"雄性":162,"无性别":29,"雌（汉语汉字）":1,"雄（中国汉字之一）":4,"雌雄同体（生物学术语）":3,"男性阿斯泰坦":1,"游戏中可选":1,"组合（汉语词语）":2,"公（汉语汉字）":14,"女性（汉语词语）":70,"男（太子）":1,"男，但人间体是女性，超越性别的存在":1,"男（雄性）":1,"可男可女，依出卷老师心情而定":1,"无":8,"男，":88,"公":4,"男1":5,"男&女（暴走士道篇）":1,"新人类1":1,"男；男（声，武神Faiz）":1,"キメラ":1,"男女共体":1,"双性":4,"女1":4,"♀":10,"雄性87.5%：雌性12.5%":1,"不明":15,"雌":13,"女，":26,"非男非女":2,"根据苇牙变换性别":1,"无性别，但偏男":1,"没有性别差异":1,"女(不包括动画)":1,"未知":4,"除林檎雨由利和黑锄文淡和蛇莓外，全为男性":1,"不定":1,"太子詹事祠部尚书司州大中正河内太守":1,"雌性或无性":1,"变性":2,"女2":1,"男3":1,"A":2,"常态为雄性，可雌雄分体":1,"男2":1,"有男有女":6,"x'b男":1,"男性":116,"软妹":2,"战士（《魔兽世界》中的战士）":1,"Man/男":1,"男（伪娘）":4,"女。":7,"：男":69,"凡人：女 神将：男":1,"男c":2,"性别不明":1,"每组女":1,"87.5% 雄性":2,"12.5% 雌性":2,"大专":1,"男女（汉语词语）":1,"： 女":4,",女":5,"男。":18,"男子":31,"男（第三季为女【配音变成了女性】）":1,"d男":1,")女":1,"人妖（社会人群）":5,"男,":9,"无（中国汉字）":14,"汉族":10,"：女":49,"： 男":8,"国（音乐火王子演唱歌曲）":2,"男性外表":1,"不详（动漫及大部分游戏）；女（《秦时明月网页游戏》）；男（真人电视剧）":2,"女（客观的讲是“雌性”）":2,"吧":3,"男n":1,"你（中国汉字）":6,"你那":9,"每组里信男":1,"壮族":1,"男（紫魅时为女）":1,"女？（男）":1,":女":4,"母猫":3,"女/♀":1,",男":10,"男♂":2,"男×男":1,"C361":1,"男男":3,":男":9,"1965年2月":1,"好玩的就喜欢":1,"母（汉字）":5,"浙江瑞安人":1,"兽族（单机游戏《魔兽争霸》中的种族）":1,"性别： 男":1,"未知（汉语词语）":14,"大者为雌":1,"身内寄存雄体":1,"BL":1,"：男性":1,"男/女":1,": 女":3,"M/M":1,"男同":1,"不详":31,"21（自然数之一）":1,"女（一开始只有纣王知道）":1,"男0":1,"伪娘（新兴词汇）":1,"仙游县（莆田市下辖县）":1,"厦门":1,"女子（汉语词语）":19,"性别（生理上的性别）":5,"安姓":3,"女ぁ男":1,"24（自然数之一）":1,"我很好，你呢（刘圣文散文）":1,"男系":1,"年（汉字释义）":6,"1987年11月22日":1,"男女各一":1,"女/男":1,"博士（研究生学位）":2,"200%女":1,"男 男":1,"女.":1,"一男一女":2,"男出处：少年漫画《家庭教师》":1,"女生（汉语词语）":9,"B型":1,"王君豪":1,"台中":1,": 男":5,"那（汉字）":10,"50% 雄性":2,"50% 雌性":2,"内蒙古赤峰":1,"丧尸（西方娱乐作品的怪物）":1,"nü":1,"female":1,"v":1,"武汉（湖北省省会）":16,"湖南益阳":1,"女孩（年轻女性）":3,"男〔出生时性别为女〕":1,"女(17岁之前是男性)":1,"男（晶晶、欢欢、迎迎）；女（贝贝、妮妮）":1,"男·":1,"性 别： 男":1,"小女孩（汉语词语）":1,"变性女":1,"女（姓氏）":2,"男（或 无）":1,"女性才有":1,"男→女":2,"多用于男性":1,"50%雄性,50%雌性":1,"男国":1,"？？？":1,"男性外观":1,"不定（爱丽丝为女性）":1,"男（本体及灵魂）→不定（由容器的性别决定）":1,"无性繁殖":2,"男（常常被误以为是女孩）":1,"女（官方一设）":3,"重生前为男，重生后为女":1,"雌性（猜测）":2,": 　男":1,"男（官方一设）":2,"通常为女性（异性同体）":1,"女性":12,"，男，":1,"87.5% 雄性，12.5% 雌性":2,"开始是男性，后被师父改造为女性":1,"女女":2,"中共党员":4,"一般为雄性":1,"男性用户为60%，女性用户为40%":2,"曹雪芹（清代作家、名著《红楼梦》的作者）":1,"20岁（歌曲）":1,"女→男":1,"母":4,"安村小美":1,"`男":4,"，男":11,"女(原男性)":1,"男/女（由玩家选择）":1,"非男":1,"民盟盟员":1,"22岁":1,"无性别（通常女性装扮）":1,"女,":2,"女性（雌性人类）":1,"半个男性":1,"男+女":1,"男（剧中有白向鸣人解释过）":1,"男，，":8,"南（汉字）":2,"Female":2,"20（自然数之一）":1,"男；有男有女（骑士钥匙）":1,"雌性（准确的说是女汉子。。。）":1,"男(漫画未连载时最初原案设定为女)":1,"每组男":1,"N女":1,"男（有变化可能性）":1,"中国":5,"人妖":1,"新人类":1,"女，学艺时曾女扮男装为司音":1,"湖北省天门县乾驿镇":1,"不限":2,"男女均有":1,"男（？）":1,"杨念恩":1,"男（伪装成女人）":1,"温柔体贴 优柔寡断":1,"李增":1,"男和女":1,"n2（氮气）":1,"中性":4,"可以任意变换性别":1,"雌雄异体":1,"雄性动物":1,"大多数是母猫":1,"字":1,"广西临桂":1,"A型":1,"广西桂平":1,"年":2,"按（动词）":2,"坤女":1,"男场上位置:后卫":1,"♂（传说中爱神和女神的代用符号）":4,"50% 雄性，50% 雌性":6,"难（难）":4,"性别":3,"不V":1,"，女":4,"男（枫）；女（南）":1,"MM":2,"nv（非易失闪存技术）":3,",男,":1,"第三性 人妖":1,"2（自然数之一）":1,"女性为主":1,"男或女":1,"1941年4月":1,"秀吉（自称男性）":1,"男(女装)":1,"未详":1,"信男最多6项最多15字":1,"。":1,"1948年9月28日男":1,"六男二女":1,"男生":1,"雌性（动画中为雄性）":2,"男足球":1,"性别不安定的状态":1,"壮":1,"不明（美版为女）":1,"甘萧省平凉县":1,"男女比例1:1":1,"男同胞":1,"雌鸽":1,"100% ♂":1,"胡光明":1,"无性":3,"棕黄色，味甘，酸苦":1,"惯用脚":1,"公熊":1,"秀吉（V.A.演唱歌曲）":1,"赵石保":1,"男（可以变成女性）":1,"25% 雄性，75% 雌性":1,"女（变性后）":1,"男 女皆有":1,"男.":1,"男士":1,"本科（学历）":1,"须眉男子":1,"女孩子":1,"男医师":1,"慢（词语）":1,"9男1女":1,"群众（汉语词语）":1,"男（外表）、男":1,"未知（零不分性别）":1,"南男":1,"阉":1,"副教授":1,"女（体）":1,"50% ♂ 50% ♀":1,"妇":2,"TS":1,"脊索动物门的一个亚门":1,"诗歌（文学体裁）":1,"汉族。":1,"河北（中华人民共和国省级行政区）":1,"乌克兰":1,"25%雄性75%雌性":1,"中":1,"少女（汉语词语）":1,"河南省伊川县":1,"临床":1,"公猫":2,"男；女（S.I.C.）":1,"大概是雌性":1,"男（人体间女）":1,"女（现实版本为男性）":1,"四川泸州":1,"男；女（舞台剧）":1,"江苏人":1,"每男":1,"雄性（他是公的哦）":1,"女20":1,"50%雄性，50%雌性":1,"无相之身":1,"小公主（美国伯内特夫人著儿童文学）":1,"十岁":1,"仙女（词语释义）":1,"调纯教爷度们娘":1,"GG":3,"伪娘":3,"恨":1,"武藏：女 小次郎：男":1,"·女":2,"♂♀":1,"女♀":1,"南宁（广西壮族自治区首府）":2,"成年男子":1,"字阿敏":1," ":1,"女美丽漂亮，成熟妩媚":1,"女湖北宜昌":1,"陈奕（中国台湾男艺人）":1,"男（曾于漫画100话首次被菈菈意外变成女生，此后又数次变成女生）":1,"21岁":1,"你说呢（Himik演唱歌曲）":1,"明明是女的又说自己是男的":1,"雌性（果宝特攻台词中用“她”称呼）":1,"酱油10克，黄酒5克，盐3克":1,"：男：":1,"云南永平":1,"原男现女":1,"女(登场被扮成男性但小茂有察觉)":1,"女（男）":1,"20":1,"男孩子":1,"人妖号":1,"多个":1,"男（基加）女（奥古玛）":1,"1966年8月":1,"，男 ，":1,"男、未知（创世王）":1,"男广东大埔":1,"泛性别":1,"待定":1,"那男":1,"女（外表为女性，实际上神是没有性别的）":1,"物业类型":1,"雌雄同体":1,"雄性大熊猫":1,"]女":1,"外观为女性":1,"男、女(性转后）":1,"男 性":1,"已出场成员均为雄性":1,"男性用户为60%，女性用户为40%。":1,"男（原本为女性）":1,"莆田线男性用户为60%女性用户40%":1,"男（人妖）":1,"女孩子ww":1,"无（机器人）":1,"男，可变身为女":1,"女生":1,"男；女（实加，试作版）；未知（试作版，先代；漫画版，先代）":1,"b女":1,"管理学学士":1,"男女皆可":1,"男，电视剧后改为女":1,"两男一女":1,"辽宁沈阳":1,"B":1,"·男":1,"保密":1,"四男二女":1,"男，后雌雄同体":1,"男．":1,"雌性精灵和雄性":1,"男（或者“伪娘”）":1,"不定（汉语词汇）":1,"帅哥人气男":1,"在《数码宝贝合体战争》中为男":1,"张椿旺":1,"雌♀":1,"严格的来说妖怪没有性别，也可以变成女人的模样":1,"男（男扮女装）":2,"男（准确地说是公）":1,"转性后为女":1,"雄性(有争议)":1,"15岁":1,"每组里信息项男最多6字，数据项最多15字":1,"为女":1,"、":1,"男女":1,"出生（词语释义）":1,"人（中国汉字）":1,"凹凸man":1,"教授（教师职称）":1,"性别：男":1,"不（汉语汉字）":1,"-":1,"虐":1,"夜鸺部队领队":1,"75% 雄性，25% 雌性":1,"主任医师":1,"湖北松滋":1,"BUFF型女神":1,"男（煞凤）/女（炎凰）":1,"男（领袖的挑战中为女）":1,"30岁":1,"n男":2,"男？":1,"田姓":1,"日本（日本国）":1,"50%雄性":1,"50%雌性":1,"腐女":1,"男（齐藤八云）、女（鞍马八云）":1,"男（推测）":1,"n":2,"孝：男 丽：女":1,"姬（经证实为男性）":1,"男 女":1,"14":1,"黑光病毒体无性别":1,"男→女（体）":1,"男(28卷64页)":1,"跟大多数耽美写手一样":1,"未知（官方公布前请不要随意修改":1,"男（公）":1,"向东":1,"足球":1,"雄性精灵":1,"女[1]":1,"女(穿越之前是男人)":1,"女（动画版未详细解释清楚）":1,"男[3]":1,"男[1]":1,"公狗":1,"不分（零不分男女，男是化身，女是附体墨夷）":1,"理御":1,"畜生道（新）为女，其余全部为男性":1,"多数为女性":1,"男（客观地说是“雄性”）":1,"男/女（电影版圣域传说）":1,"男（番外6中变成因女神宝物成了萝莉）":1,"世羽（女）":1,"男(外观决定)":1,"男（黑无常·常昊灵），女（白无常·常宣灵）":1,"可能为雄性":1,"菩萨身，不分男女":1,"另类":1,"男（召唤者不分）":1,"女【灵虫本来无性别，通过修炼可以选择成为男性或女性，后糖宝选择成为女孩】":1,"男（原著小说里为男性，漫画里为女性）":1,"男\\女":1,"男性和女性":1,"4男2女":1,"女性（仅以声音辨别，电脑是无性别的）":1}
			docs = {'男':2, '男和女':1, '男·':1, '性 别： 男':1,"女孩":1, "雄":33,"雄性":162}
		word_piece_dict = {}
		pa = r'\d+(\.\d+%{0,1}){0,1}'
		new_docs = {}
		cand_words = set()
		for q, info in docs.items():
			new_docs[q] = []
			temp_q = re.sub(pa, '', q)
			temp_qs = temp_q.lower().split(' ')
			for temp_q in temp_qs:
				if temp_q in cand_words:
					new_docs[q].append(temp_q) 
					continue

				start_idx = 0
				for i in range(len(temp_q)):
					if temp_q[i] in self.puncs:
						piece = temp_q[start_idx:i]
						if piece:
							cand_words.add(piece)
							new_docs[q].append(piece)
						start_idx = i+1
					
				if start_idx < len(temp_q):
					piece = temp_q[start_idx:]
					cand_words.add(piece)
					new_docs[q].append(piece)


		word_pieces = []
		word_piece_dict = {}
		# print("cand_words:", json.dumps(list(cand_words), ensure_ascii=False))
		# print("new_docs:", json.dumps(new_docs, ensure_ascii=False))

		for q, pieces in new_docs.items():

			all_cand_piece = []
			for piece in pieces:
				in_pieces = [] 
				if piece in self.piece4word:
					all_cand_piece.extend(self.piece4word[piece])
					continue

				for w in cand_words:
					if w in piece:
						in_pieces.append(w)
				self.piece4word[piece] = in_pieces
				all_cand_piece.extend(in_pieces)
							
			word_pieces.extend(all_cand_piece)
			word_piece_dict[q] = Counter(all_cand_piece)

		# print("word_piece_dict:", json.dumps(word_piece_dict, ensure_ascii=False))
		
		# print('所有的n-gram:')
		# print(json.dumps(word_piece_dict))
		wp_count = Counter(word_pieces)
		
		return wp_count, word_piece_dict	

	def get_char_piece_freq_wo_freq(self, docs = None):
		#Gets the frequency of the char piece level, regardless of how often each value originally occurs
		if docs is None:
			docs = self.docs
		word_pieces = []
		word_piece_dict = {}
		for q in docs:
			# temp_q = q.split('#****#')[0]
			temp_qs = q.split(' ')
			temp_piece = []
			for temp_q in temp_qs:
				if temp_q in self.piece4word:
					temp_piece.extend(self.piece4word[temp_q])
					continue

				pieces = []	
				for i in range(len(temp_q)):
					if temp_q[i] in self.puncs:
						continue
					for j in range(i+1, len(temp_q)+1):
						if temp_q[j-1] in self.puncs:
							break
						w_piece = temp_q[i:j]
						temp_piece.append(w_piece)
						pieces.append(w_piece)
				if pieces:
					self.piece4word[temp_q] = pieces

			word_pieces.extend(temp_piece)
			word_piece_dict[q] = Counter(temp_piece)	
			
		# print('所有的n-gram:', word_piece_dict)
		wp_count = Counter(word_pieces)

		return wp_count, word_piece_dict

	def get_char_piece_freq_w_freq(self, docs=None):
		#Gets the frequency of the char piece level, considering how often each value occurs, man:m,a,n,ma,an,man
		if docs is None:
			docs = {"v_total":325501,"男":249397,"男性用户约占61%，女性约39%":1,"北辰社":1,"女（汉字）":63644,"女":1,"个体不同":1,"雌性":78,"男（部分章节性别为女）":1,"雄":33,"雄性":162,"无性别":29,"雌（汉语汉字）":1,"雄（中国汉字之一）":4,"雌雄同体（生物学术语）":3,"男性阿斯泰坦":1,"游戏中可选":1,"组合（汉语词语）":2,"公（汉语汉字）":14,"女性（汉语词语）":70,"男（太子）":1,"男，但人间体是女性，超越性别的存在":1,"男（雄性）":1,"可男可女，依出卷老师心情而定":1,"无":8,"男，":88,"公":4,"男1":5,"男&女（暴走士道篇）":1,"新人类1":1,"男；男（声，武神Faiz）":1,"キメラ":1,"男女共体":1,"双性":4,"女1":4,"♀":10,"雄性87.5%：雌性12.5%":1,"不明":15,"雌":13,"女，":26,"非男非女":2,"根据苇牙变换性别":1,"无性别，但偏男":1,"没有性别差异":1,"女(不包括动画)":1,"未知":4,"除林檎雨由利和黑锄文淡和蛇莓外，全为男性":1,"不定":1,"太子詹事祠部尚书司州大中正河内太守":1,"雌性或无性":1,"变性":2,"女2":1,"男3":1,"A":2,"常态为雄性，可雌雄分体":1,"男2":1,"有男有女":6,"x'b男":1,"男性":116,"软妹":2,"战士（《魔兽世界》中的战士）":1,"Man/男":1,"男（伪娘）":4,"女。":7,"：男":69,"凡人：女 神将：男":1,"男c":2,"性别不明":1,"每组女":1,"87.5% 雄性":2,"12.5% 雌性":2,"大专":1,"男女（汉语词语）":1,"： 女":4,",女":5,"男。":18,"男子":31,"男（第三季为女【配音变成了女性】）":1,"d男":1,")女":1,"人妖（社会人群）":5,"男,":9,"无（中国汉字）":14,"汉族":10,"：女":49,"： 男":8,"国（音乐火王子演唱歌曲）":2,"男性外表":1,"不详（动漫及大部分游戏）；女（《秦时明月网页游戏》）；男（真人电视剧）":2,"女（客观的讲是“雌性”）":2,"吧":3,"男n":1,"你（中国汉字）":6,"你那":9,"每组里信男":1,"壮族":1,"男（紫魅时为女）":1,"女？（男）":1,":女":4,"母猫":3,"女/♀":1,",男":10,"男♂":2,"男×男":1,"C361":1,"男男":3,":男":9,"1965年2月":1,"好玩的就喜欢":1,"母（汉字）":5,"浙江瑞安人":1,"兽族（单机游戏《魔兽争霸》中的种族）":1,"性别： 男":1,"未知（汉语词语）":14,"大者为雌":1,"身内寄存雄体":1,"BL":1,"：男性":1,"男/女":1,": 女":3,"M/M":1,"男同":1,"不详":31,"21（自然数之一）":1,"女（一开始只有纣王知道）":1,"男0":1,"伪娘（新兴词汇）":1,"仙游县（莆田市下辖县）":1,"厦门":1,"女子（汉语词语）":19,"性别（生理上的性别）":5,"安姓":3,"女ぁ男":1,"24（自然数之一）":1,"我很好，你呢（刘圣文散文）":1,"男系":1,"年（汉字释义）":6,"1987年11月22日":1,"男女各一":1,"女/男":1,"博士（研究生学位）":2,"200%女":1,"男 男":1,"女.":1,"一男一女":2,"男出处：少年漫画《家庭教师》":1,"女生（汉语词语）":9,"B型":1,"王君豪":1,"台中":1,": 男":5,"那（汉字）":10,"50% 雄性":2,"50% 雌性":2,"内蒙古赤峰":1,"丧尸（西方娱乐作品的怪物）":1,"nü":1,"female":1,"v":1,"武汉（湖北省省会）":16,"湖南益阳":1,"女孩（年轻女性）":3,"男〔出生时性别为女〕":1,"女(17岁之前是男性)":1,"男（晶晶、欢欢、迎迎）；女（贝贝、妮妮）":1,"男·":1,"性 别： 男":1,"小女孩（汉语词语）":1,"变性女":1,"女（姓氏）":2,"男（或 无）":1,"女性才有":1,"男→女":2,"多用于男性":1,"50%雄性,50%雌性":1,"男国":1,"？？？":1,"男性外观":1,"不定（爱丽丝为女性）":1,"男（本体及灵魂）→不定（由容器的性别决定）":1,"无性繁殖":2,"男（常常被误以为是女孩）":1,"女（官方一设）":3,"重生前为男，重生后为女":1,"雌性（猜测）":2,": 　男":1,"男（官方一设）":2,"通常为女性（异性同体）":1,"女性":12,"，男，":1,"87.5% 雄性，12.5% 雌性":2,"开始是男性，后被师父改造为女性":1,"女女":2,"中共党员":4,"一般为雄性":1,"男性用户为60%，女性用户为40%":2,"曹雪芹（清代作家、名著《红楼梦》的作者）":1,"20岁（歌曲）":1,"女→男":1,"母":4,"安村小美":1,"`男":4,"，男":11,"女(原男性)":1,"男/女（由玩家选择）":1,"非男":1,"民盟盟员":1,"22岁":1,"无性别（通常女性装扮）":1,"女,":2,"女性（雌性人类）":1,"半个男性":1,"男+女":1,"男（剧中有白向鸣人解释过）":1,"男，，":8,"南（汉字）":2,"Female":2,"20（自然数之一）":1,"男；有男有女（骑士钥匙）":1,"雌性（准确的说是女汉子。。。）":1,"男(漫画未连载时最初原案设定为女)":1,"每组男":1,"N女":1,"男（有变化可能性）":1,"中国":5,"人妖":1,"新人类":1,"女，学艺时曾女扮男装为司音":1,"湖北省天门县乾驿镇":1,"不限":2,"男女均有":1,"男（？）":1,"杨念恩":1,"男（伪装成女人）":1,"温柔体贴 优柔寡断":1,"李增":1,"男和女":1,"n2（氮气）":1,"中性":4,"可以任意变换性别":1,"雌雄异体":1,"雄性动物":1,"大多数是母猫":1,"字":1,"广西临桂":1,"A型":1,"广西桂平":1,"年":2,"按（动词）":2,"坤女":1,"男场上位置:后卫":1,"♂（传说中爱神和女神的代用符号）":4,"50% 雄性，50% 雌性":6,"难（难）":4,"性别":3,"不V":1,"，女":4,"男（枫）；女（南）":1,"MM":2,"nv（非易失闪存技术）":3,",男,":1,"第三性 人妖":1,"2（自然数之一）":1,"女性为主":1,"男或女":1,"1941年4月":1,"秀吉（自称男性）":1,"男(女装)":1,"未详":1,"信男最多6项最多15字":1,"。":1,"1948年9月28日男":1,"六男二女":1,"男生":1,"雌性（动画中为雄性）":2,"男足球":1,"性别不安定的状态":1,"壮":1,"不明（美版为女）":1,"甘萧省平凉县":1,"男女比例1:1":1,"男同胞":1,"雌鸽":1,"100% ♂":1,"胡光明":1,"无性":3,"棕黄色，味甘，酸苦":1,"惯用脚":1,"公熊":1,"秀吉（V.A.演唱歌曲）":1,"赵石保":1,"男（可以变成女性）":1,"25% 雄性，75% 雌性":1,"女（变性后）":1,"男 女皆有":1,"男.":1,"男士":1,"本科（学历）":1,"须眉男子":1,"女孩子":1,"男医师":1,"慢（词语）":1,"9男1女":1,"群众（汉语词语）":1,"男（外表）、男":1,"未知（零不分性别）":1,"南男":1,"阉":1,"副教授":1,"女（体）":1,"50% ♂ 50% ♀":1,"妇":2,"TS":1,"脊索动物门的一个亚门":1,"诗歌（文学体裁）":1,"汉族。":1,"河北（中华人民共和国省级行政区）":1,"乌克兰":1,"25%雄性75%雌性":1,"中":1,"少女（汉语词语）":1,"河南省伊川县":1,"临床":1,"公猫":2,"男；女（S.I.C.）":1,"大概是雌性":1,"男（人体间女）":1,"女（现实版本为男性）":1,"四川泸州":1,"男；女（舞台剧）":1,"江苏人":1,"每男":1,"雄性（他是公的哦）":1,"女20":1,"50%雄性，50%雌性":1,"无相之身":1,"小公主（美国伯内特夫人著儿童文学）":1,"十岁":1,"仙女（词语释义）":1,"调纯教爷度们娘":1,"GG":3,"伪娘":3,"恨":1,"武藏：女 小次郎：男":1,"·女":2,"♂♀":1,"女♀":1,"南宁（广西壮族自治区首府）":2,"成年男子":1,"字阿敏":1," ":1,"女美丽漂亮，成熟妩媚":1,"女湖北宜昌":1,"陈奕（中国台湾男艺人）":1,"男（曾于漫画100话首次被菈菈意外变成女生，此后又数次变成女生）":1,"21岁":1,"你说呢（Himik演唱歌曲）":1,"明明是女的又说自己是男的":1,"雌性（果宝特攻台词中用“她”称呼）":1,"酱油10克，黄酒5克，盐3克":1,"：男：":1,"云南永平":1,"原男现女":1,"女(登场被扮成男性但小茂有察觉)":1,"女（男）":1,"20":1,"男孩子":1,"人妖号":1,"多个":1,"男（基加）女（奥古玛）":1,"1966年8月":1,"，男 ，":1,"男、未知（创世王）":1,"男广东大埔":1,"泛性别":1,"待定":1,"那男":1,"女（外表为女性，实际上神是没有性别的）":1,"物业类型":1,"雌雄同体":1,"雄性大熊猫":1,"]女":1,"外观为女性":1,"男、女(性转后）":1,"男 性":1,"已出场成员均为雄性":1,"男性用户为60%，女性用户为40%。":1,"男（原本为女性）":1,"莆田线男性用户为60%女性用户40%":1,"男（人妖）":1,"女孩子ww":1,"无（机器人）":1,"男，可变身为女":1,"女生":1,"男；女（实加，试作版）；未知（试作版，先代；漫画版，先代）":1,"b女":1,"管理学学士":1,"男女皆可":1,"男，电视剧后改为女":1,"两男一女":1,"辽宁沈阳":1,"B":1,"·男":1,"保密":1,"四男二女":1,"男，后雌雄同体":1,"男．":1,"雌性精灵和雄性":1,"男（或者“伪娘”）":1,"不定（汉语词汇）":1,"帅哥人气男":1,"在《数码宝贝合体战争》中为男":1,"张椿旺":1,"雌♀":1,"严格的来说妖怪没有性别，也可以变成女人的模样":1,"男（男扮女装）":2,"男（准确地说是公）":1,"转性后为女":1,"雄性(有争议)":1,"15岁":1,"每组里信息项男最多6字，数据项最多15字":1,"为女":1,"、":1,"男女":1,"出生（词语释义）":1,"人（中国汉字）":1,"凹凸man":1,"教授（教师职称）":1,"性别：男":1,"不（汉语汉字）":1,"-":1,"虐":1,"夜鸺部队领队":1,"75% 雄性，25% 雌性":1,"主任医师":1,"湖北松滋":1,"BUFF型女神":1,"男（煞凤）/女（炎凰）":1,"男（领袖的挑战中为女）":1,"30岁":1,"n男":2,"男？":1,"田姓":1,"日本（日本国）":1,"50%雄性":1,"50%雌性":1,"腐女":1,"男（齐藤八云）、女（鞍马八云）":1,"男（推测）":1,"n":2,"孝：男 丽：女":1,"姬（经证实为男性）":1,"男 女":1,"14":1,"黑光病毒体无性别":1,"男→女（体）":1,"男(28卷64页)":1,"跟大多数耽美写手一样":1,"未知（官方公布前请不要随意修改":1,"男（公）":1,"向东":1,"足球":1,"雄性精灵":1,"女[1]":1,"女(穿越之前是男人)":1,"女（动画版未详细解释清楚）":1,"男[3]":1,"男[1]":1,"公狗":1,"不分（零不分男女，男是化身，女是附体墨夷）":1,"理御":1,"畜生道（新）为女，其余全部为男性":1,"多数为女性":1,"男（客观地说是“雄性”）":1,"男/女（电影版圣域传说）":1,"男（番外6中变成因女神宝物成了萝莉）":1,"世羽（女）":1,"男(外观决定)":1,"男（黑无常·常昊灵），女（白无常·常宣灵）":1,"可能为雄性":1,"菩萨身，不分男女":1,"另类":1,"男（召唤者不分）":1,"女【灵虫本来无性别，通过修炼可以选择成为男性或女性，后糖宝选择成为女孩】":1,"男（原著小说里为男性，漫画里为女性）":1,"男\\女":1,"男性和女性":1,"4男2女":1,"女性（仅以声音辨别，电脑是无性别的）":1}
		
		word_pieces = []
		word_piece_dict = {}
		for q, info in docs.items():
			# temp_q = q.split('#****#')[0]
			temp_qs = q.lower().split(' ')
			if len(info) == 0:
				freq = info
			else:
				freq = info[-1]  #默认最后一列保存的是value出现的频次

			temp_piece = []
			for temp_q in temp_qs:
				if temp_q in self.piece4word:
					temp_piece.extend(self.piece4word[temp_q]*freq)
					continue

				pieces = []
				for i in range(len(temp_q)):
					if temp_q[i] in self.puncs:
						continue
					for j in range(i+1, len(temp_q)+1):
						if temp_q[j-1] in self.puncs:
							break
						w_piece = temp_q[i:j]
						temp_piece.extend([w_piece]*freq)
						pieces.append(w_piece)
				if pieces:
					self.piece4word[temp_q] = pieces
					# print(temp_q, pieces)
							
			word_pieces.extend(temp_piece)
			word_piece_dict[q] = Counter(temp_piece)
		
		# print('所有的n-gram:')
		# print(json.dumps(word_piece_dict))
		wp_count = Counter(word_pieces)
		
		return wp_count, word_piece_dict	

	def get_tfidf4wp(self, word_piece_dict, wp_count, docs=None):
		#get tfidf for word piece for each value

		doc_total = 0.0   
		dc_wp_dict = defaultdict(int)  
		tf_dict = {}
		#计算tf
		for q, info in word_piece_dict.items():
			temp_c = docs.get(q, [1])[-1]
			doc_total += temp_c
			count = 0.0
			tf_dict[q] = {}
			for w, c in info.items():
				dc_wp_dict[w] += temp_c
				count += c
			for w, c in info.items():
				tf_dict[q][w] = round(c*1.0/count, 4)

		#idf
		idf_dict = {}
		for w, c in dc_wp_dict.items():
			idf_dict[w] = round(math.log(doc_total/c, 10),4)	

		#tf-idf
		tfidf_dict = {}
		for q, info in tf_dict.items():
			tfidf_dict[q] = {}
			for w, tf in info.items():
				tfidf_dict[q][w] = round(tf*idf_dict[w], 4)

		if self.idf_fw:	
			self.idf_fw.write(json.dumps(tfidf_dict, ensure_ascii=False)+'\n')

		return tfidf_dict

	def get_idf4wp(self, word_piece_dict, wp_count, docs=None):
		#get idf for word piece for each value

		doc_total = 0.0   
		dc_wp_dict = defaultdict(int)  
		q_idf_dict = {}

		for q, info in word_piece_dict.items():
			q_idf_dict[q] = {}
			temp_c = docs.get(q, [1])[-1]
			doc_total += temp_c
			count = 0.0
			for w, c in info.items():
				dc_wp_dict[w] += temp_c
				count += c
				q_idf_dict[q][w] = 0

		#idf
		idf_dict = {}
		for w, c in dc_wp_dict.items():
			idf_dict[w] = round(math.log(doc_total/c, 10),4)	

		for q, info in q_idf_dict.items():
			for w, c in info.items():
				q_idf_dict[q][w] = idf_dict[w]


		if self.idf_fw:	
			self.idf_fw.write(json.dumps(q_idf_dict, ensure_ascii=False)+'\n')

		return q_idf_dict

	def get_sim_by_tfidf(self, tfidf_dict):
		#get similarity by tfidf
		sims_dict = {}
		min_sim = 1.0
		max_sim = 0.0 
		docs = list(tfidf_dict.keys())
		for i, q1 in enumerate(docs):
			for q2 in docs[i+1:]:
				wp_1 = tfidf_dict[q1]
				wp_2 = tfidf_dict[q2]
				try:
					all_ps = list(set(wp_1.keys()) | set(wp_2.keys()))
					f1 = np.array([wp_1.get(x, 0) for x in all_ps])
					f2 = np.array([wp_2.get(x, 0) for x in all_ps])
					sim = round(cosine_similarity([f1], [f2])[0][0], 4)
					if sim > 1.0:
						print(q1, q2, sim, 'hypersimilarity')
					elif sim == 1.0:
						print(q1, q2, sim, 'similarity == 1')
					sims_dict[(q1, q2)] = sim
					max_sim = max(max_sim, sim)
					min_sim = min(min_sim, sim)
				except:
					traceback.print_exc()
					print(q1, wp_1, q2, wp_2, 'failure similarity')
 
		# print("freq_sims_dict:", sorted(sims_dict.items(), key=lambda x:x[1], reverse=True))		
		print('max similarity', max_sim)
		print('min similarity', min_sim)		
		return sims_dict		

	def get_sim_by_wordpiece(self, wp_count, word_piece_dict):
		sims_dict = {}
		min_sim = 1.0
		max_sim = 0.0
		docs = list(word_piece_dict.keys())
		for i, q1 in enumerate(docs):
			for q2 in docs[i+1:]:
				wp_1 = word_piece_dict[q1].keys()
				wp_2 = word_piece_dict[q2].keys()
				inter = set([x for x in wp_1 if x in wp_2])  
				union = list(wp_1 | wp_2)              
				fenzi = 0
				for x in inter:
					fenzi += wp_count[x]
				fenmu = 0 
				for x in union:
					fenmu += wp_count[x]
				sim = round(fenzi*1.0/fenmu, 4)
				# print(inter, union, fenzi, fenmu, sim)
				if sim > 1.0:
					print(q1, q2, sim, fenzi, fenmu, wp_1, wp_2, 'hypersimilarity')
				elif sim == 1.0:
					print(q1, q2, sim, fenzi, fenmu, wp_1, wp_2, 'similarity == 1')
				sims_dict[(q1, q2)] = sim
				max_sim = max(max_sim, sim)
				min_sim = min(min_sim, sim)
 
		# print("freq_sims_dict:", sorted(sims_dict.items(), key=lambda x:x[1], reverse=True))		
		print('max similarity', max_sim)
		print('min similarity', min_sim)		
		return sims_dict

	def readembs(self, file):
		with open(file, 'r', encoding='utf8') as fr:
			reader = csv.reader(fr)
			for row in reader:
				# print(row)
				name, emb = row
				emb = json.loads(emb)
				# print(name)
				# print(len(emb))
				# print(emb[0], emb[-1])
				self.all_embs[name] = emb
		print('the number of values:', len(self.all_embs.keys()))

	def get_embs(self, docs):
		target_embs = []
		for x in docs:
			target_embs.append(self.all_embs[x])

		return target_embs

	def get_sim_by_embs(self, docs):
		sims_dict = {}
		min_sim = 1.0
		max_sim = 0.0 
		if isinstance(docs, dict):
			docs = list(docs.keys())

		for i, q1 in enumerate(docs):
			for q2 in docs[i+1:]:
				try:
					emb_1 = np.array(self.all_embs[q1])
					emb_2 = np.array(self.all_embs[q2])
				
					sim = round(cosine_similarity([emb_1], [emb_2])[0][0], 4)
					if sim > 1.0:
						print(q1, q2, sim, 'hypersimilarity')
					elif sim == 1.0:
						print(q1, q2, sim, 'similarity == 1')
					sims_dict[(q1, q2)] = sim
					max_sim = max(max_sim, sim)
					min_sim = min(min_sim, sim)
				except:
					traceback.print_exc()
					print(q1, q2, 'failure similarity')
		return sims_dict


	def merge_sims(self, sims1, sims2, alpha = 1.0):
		# merge two level similarities
		#alphas is the rate of sims1
		if alpha == 1.0:
			return sims1
		elif alpha == 0.0:
			return sims2
		for (v1, v2), sim1 in sims1.items():
			sim2 = sims2[(v1, v2)]
			sims1[(v1, v2)] = sim1* alpha + (1 - alpha)*sim2

		return sims1
			
	def cal_sim_by_freq(self, docs=None):
		if self.sim_type == 'cp_w_freq':
			wp_count, word_piece_dict = self.get_char_piece_freq_w_freq(docs)
			sim_pairs = self.get_sim_by_wordpiece(wp_count, word_piece_dict)

		elif self.sim_type == 'cp_wo_freq':
			wp_count, word_piece_dict = self.get_char_piece_freq_wo_freq(docs)
			sim_pairs = self.get_sim_by_wordpiece(wp_count, word_piece_dict)

		elif self.sim_type == 'wp_tfidf':
			wp_count, word_piece_dict = self.get_word_piece_freq_wo_freq(docs)
			tfidf_dict = self.get_tfidf4wp(word_piece_dict, wp_count, docs)
			sim_pairs = self.get_sim_by_tfidf(tfidf_dict)

		elif self.sim_type == 'wp_idf':
			wp_count, word_piece_dict = self.get_word_piece_freq_wo_freq(docs)
			idf_dict = self.get_idf4wp(word_piece_dict, wp_count, docs)
			sim_pairs = self.get_sim_by_tfidf(idf_dict)

		elif 'text_emb' in self.sim_type:
			# text_embs = self.get_embs(docs)
			sim_pairs = self.get_sim_by_embs(docs)
			if self.sim_type[-1] in "0123456789":
				alpha = int(self.sim_type[-1]) *1.0 / 10
			else:
				alpha = 1.0
			print('alpha:', alpha)
			if alpha != 1.0:
				if 'cp_w_f' in self.sim_type:
					wp_count, word_piece_dict = self.get_char_piece_freq_w_freq(docs)
					temp_sim_pairs = self.get_sim_by_wordpiece(wp_count, word_piece_dict)
					sim_pairs = self.merge_sims(sim_pairs, temp_sim_pairs, alpha)

				elif 'cp_wo_f' in self.sim_type:
					wp_count, word_piece_dict = self.get_char_piece_freq_wo_freq(docs)
					temp_sim_pairs = self.get_sim_by_wordpiece(wp_count, word_piece_dict)
					sim_pairs = self.merge_sims(sim_pairs, temp_sim_pairs, alpha)

				# elif 'wp_tfidf' in self.sim_type:
				elif 'tfidf' in self.sim_type:
					wp_count, word_piece_dict = self.get_word_piece_freq_wo_freq(docs)
					tfidf_dict = self.get_tfidf4wp(word_piece_dict, wp_count, docs)
					temp_sim_pairs = self.get_sim_by_tfidf(tfidf_dict)
					sim_pairs = self.merge_sims(sim_pairs, temp_sim_pairs, alpha)

				elif 'wp_idf' in self.sim_type:
					wp_count, word_piece_dict = self.get_word_piece_freq_wo_freq(docs)
					tfidf_dict = self.get_idf4wp(word_piece_dict, wp_count, docs)
					temp_sim_pairs = self.get_sim_by_tfidf(tfidf_dict)
					sim_pairs = self.merge_sims(sim_pairs, temp_sim_pairs, alpha)

				else:
					print('unconsidered case sim_type', self.sim_type)

		else:
			print('unconsidered case sim_type', sim_type)

		if self.wp_fw:	
			self.wp_fw.write(json.dumps(word_piece_dict, ensure_ascii=False)+'\n')
			
		
		return sim_pairs

	def cal_sim_by_emb(self, docs=None):
		return '' 	

class Cluster_dis:
	def __init__(self, clu_type = 'dbscan_text_emb_large'):
		self.clu_type = clu_type

		self.Sim_cal = Sim_cal(sim_type=clu_type)

	def get_matrix(self, docs, sim_pairs):
		if 'emb_large' in self.clu_type:
			matrix = self.get_matrix_by_embs(docs)

		elif 'sim' in self.clu_type:
			matrix = self.get_matrix_by_sims(docs, sim_pairs)
		
		else:
			print('Unconsidered matrix construction', self.clu_type)

		return matrix

	def get_matrix_by_embs(self, docs):
		embs = self.Sim_cal.get_embs(docs)
		return embs

	def get_matrix_by_sims(self, docs, sims):
		pass


	def dbscan(self, matrix, docs):
		"""
		eps:Represents the radius, the larger the number of categories the smaller
		"""
		dbscan = DBSCAN(eps=0.5, min_samples=2)
		y_pred = dbscan.fit_predict(matrix)

		#后处理
		res = {}
		for i, clus in enumerate(y_pred):
			if clus not in res:
				res[clus] = []
			try:
				res[clus].append(docs[i])
			except:
				traceback.print_exc()
				print(clus, i, len(docs))

		return list(res.values())

	def kmeans(self, matrix, docs, n_clusters=10):
		max_cs = len(docs)
		k_range = range(1, max_cs+1)
		inertias  = []   
		temp_res = {}
		for n_clusters in k_range:
			kmeans = KMeans(n_clusters =n_clusters)
			kmeans.fit(matrix)

			centroids = kmeans.cluster_centers_  
			labels = kmeans.labels_
			inertias.append(kmeans.inertia_)  

			#后处理
			res = {}
			for i, clus in enumerate(labels):
				if clus not in res:
					res[clus] = []
				res[clus].append(docs[i])

			temp_res[n_clusters] = res

		inertia_diff = np.diff(inertias)
		print('inertias:', inertias)
		print("inertia_diff:", inertia_diff)
		k_elbow = k_range[-1]
		max_diff = 0
		for i in range(len(inertia_diff)):
			if abs(inertia_diff[i]) > max_diff:
				max_diff = abs(inertia_diff[i])
				if inertia_diff[i] < 0:   
					k_elbow = i+2
				else:
					k_elbow = i+1


		res = temp_res[k_elbow]


		return list(res.values())

	def get_clusters(self, docs, sim_pairs=None, n_cluster=10):
		matrix = self.get_matrix(docs, sim_pairs)
		if 'dbscan' in self.clu_type :
			clusters = self.dbscan(matrix, docs)
		elif 'kmeans' in self.clu_type:
			clusters = self.kmeans(matrix, docs, n_cluster)
		else:
			print('unconsidered cluster methods', self.clu_type)

		return clusters 


class Comunity_dis:
	#社区发现算法
	def __init__(self, sim_type = 'cp_w_freq', com_type = None, wp_fw=None, sim_fw=None, idf_fw=None, thre=0.7):
		data = {"女":["妇女","女女","女性","女","女【灵虫本来无性别，通过修炼可以选择成为男性或女性，后糖宝选择成为女孩】","女（动画版未详细解释清楚）","女（体）","雌《女》","性别：女","男→女（体）","女(17岁之前是男性)","女性（仅以声音辨别，电脑是无性别的）","女（官方一设）","女（汉字）","女{主人格","雌性（准确的说是女汉子。。。）","男；女（S.I.C.）","女性(大神暗示过，大神传确认)","男（曾于漫画100话首次被菈菈意外变成女生，此后又数次变成女生）","腐女","女(登场被扮成男性但小茂有察觉)","女子（汉语词语）","女性（雌性人类）","女（外表为女性，实际上神是没有性别的）","N女","b女","应为女","坤女","女性（汉语词语）","女（无性别，附在一村妇体内）","200%女","多数为女性","中老年女性","外观为女性","转性后为女","女孩子ww","女（姓氏）","富家女","女20","女(原男性)","女孩子","女（一开始只有纣王知道）","女，学艺时曾女扮男装为司音","女孩（年轻女性）","女(不包括动画)","少女（汉语词语）","女【灵虫本来无性别，通过修炼可以选择成为男性或女性，后糖宝选择成为女孩】","女（客观的讲是“雌性”）","开始是男性，后被师父改造为女性","原男现女","女生（汉语词语）","女（有争议）","小女孩（汉语词语）","﻿女","女1","女2","女♀","为女","女生","女（好像也没什么不对......）","灵族男或女","不明（美版为女）","每组女","男、女(性转后）","BUFF型女神","男女（汉语词语）","通常为女性（异性同体）"],"男/女":["男/女","女/♀"],"男":["男系","　男","男．","`男","男","男生","男（本体及灵魂）→不定（由容器的性别决定）","男；男（声，武神Faiz）","男出处：少年漫画《家庭教师》","在《数码宝贝合体战争》中为男","男（剧中有白向鸣人解释过）","男人（具有xy染色体的人）","男（兽人）","男（官方一季设定）","男（由原始程序而定）","男(28卷64页)","男（准确地说是公）","男士","男？","那男","男n","男♂","男子","男性","男同","男国","男1","男（公）","男性外观","均为男性","须眉男子","男性角色","通常为男","成年男子","Man/男","男（？）","男（雄）","x'b男","男性外表","性别：男","男（或无）","1948年9月28日男","男(游戏内可自由选择性别)","这么可爱一定是男孩子","男（客观地说是“雄性”）","男孩（年轻的男性人类）","男2","男0","男（召唤者不分）","男场上位置:后卫","秀吉（自称男性）","剧情中默认为男","男（官方一季设）","统统是男的","帅哥人气男","男（太子）","男广东大埔","男（已知）","男（推测）","男（公？）","男祖","男（雄性）","男性或中性","推测为男性","南男","男3"],"每男":["男c","每男"],"d男":["男同胞","男×男","男医师","每组男","n男","男足球","均为男","男孩子","d男","男/男"],"每组里信男":["每组里信男","男；男（外表）","男性阿斯泰坦","男（官方一设）","男（进之介）","男(外观决定)","男（外表）、男"],"母猫":["母（汉字）","母虎","母猫"],"母（公蚊吃素，母蚊吃荤--不一定，公的也會吸血）":["雌鸽","母（公蚊吃素，母蚊吃荤--不一定，公的也會吸血）","母","雌性（果宝特攻台词中用“她”称呼）","雌性","雌","雌♀","雌体"],"雄":["雄（中国汉字之一）","雄","雄性","已出场成员均为雄性","雄性动物","雄性(有争议)","起源为雄性","雄性大熊猫","一般为雄性","雄性机械羊","可能为雄性","100%雄性"],"雄性2":["雄性精灵","雄性2","雄（中国汉字之一）"],"无性":["无性别","无性"],"MM":["M/M","MM"],"未知（官方公布前请不要随意修改":["未知","未知（官方公布前请不要随意修改"],"伪娘":["男（或者“伪娘”）","伪娘（新兴词汇）","男（伪娘）","伪娘","冒失娘"],"至今不明":["至今不明","性别不明"],"不确定（定义）":["性别不安定的状态","不确定（定义）"],"男（有变化可能性）":["男（有变化可能性）","男（可以变成女性）"],"male":["male","♂1:♀1","凹凸man","GG"],"Female":["Female","female"],"男女皆可":["男女皆可","男、女(性转后）","菩萨身（不分男女）","菩萨身，不分男女","男女同体","女【灵虫本来无性别，通过修炼可以选择成为男性或女性，后糖宝选择成为女孩】","非男非女"],"男（男扮女装）":["男（男扮女装）","男(漫画未连载时最初原案设定为女)","男（常常被误以为是女孩）"],"男（原本为女性）":["男（原本为女性）","男〔出生时性别为女〕","男（常常被误以为是女孩）"],"母":["母","大多数是母猫","可爱的小母猫"],"雌":["雌","雌性或无性","雌性（果宝特攻台词中用“她”称呼）","大部分是雌","大概是雌性","100%雌性","雌（汉语汉字）","雌性（猜测）"],"雌雄同体":["雌雄同体","雌雄同体（生物学术语）"],"公":["公","公（汉语汉字）","雄性（他是公的哦）"],"变性":["变性","女（变性后）"],"noadded_vs":["非男","雌性（动画中为雄性）","♂♀","男（齐藤八云）、女（鞍马八云）","100%♂","男，但人间体是女性，超越性别的存在","阉","严格的来说妖怪没有性别，也可以变成女人的模样","男（原著小说里为男性，漫画里为女性）","无","不定","nan","变性女","不限","无性别，但偏男","第三性人妖","50%♂50%♀","男&女（暴走士道篇）","男（部分章节性别为女）","男孩和女孩","男（紫魅时为女）","男（番外6中变成因女神宝物成了萝莉）","女（男）","女（现实版本为男性）","未详","不明（推测为男性）","男和女","未知（零不分性别）","水母雌雄同体","不分（零不分男女，男是化身，女是附体墨夷）","女(穿越之前是男人)","女/男","男→女","大者为雌","男，电视剧后改为女","男女共体","男，可变身为女","男（人体间女）","妇","不详","男、未知（创世王）","男性和女性","♂（传说中爱神和女神的代用符号）","明明是女的又说自己是男的","不明","重生前为男，重生后为女","公熊","男（人妖）","不定（汉语词汇）","男（第三季为女【配音变成了女性】）","常态为雄性，可雌雄分体","软妹","女→男","小公主（美国伯内特夫人著儿童文学）","男（伪装成女人）","男（可捏造成女性形态）","公猫","公狗","未知（汉语词语）","男男","男女","男（领袖的挑战中为女）","男性用户为60%，女性用户为40%","无性别（通常女性装扮）","公鸡（公鸡）","nü","双性"]}
		self.docs = []
		for k, vs in data.items():
			self.docs.extend(vs)
		
		self.com_type = com_type
		self.Sim_cal = Sim_cal(sim_type = sim_type, wp_fw= wp_fw, idf_fw=idf_fw)
		self.sim_fw = sim_fw
		self.thre = thre
		self.sim_type = sim_type

		self.translate = {}  

	def delete_trans(self, docs):
		new_docs = {}
		for q, freq in docs.items():
			temp_q = q.split('#****#')[0] 
			new_docs[temp_q] = [[q], freq]
			self.translate[temp_q] = q
		return new_docs

	def add_trans(self, docs):
		temp_docs = []
		for q in docs:
			temp_docs.append(self.translate[q])
		return temp_docs

	def create_graph(self, docs=None, freq_sim_dict=None, sims2 = None):
		if freq_sim_dict is None:
			if docs is None:
				docs = self.docs

			if dataset == 'dbpedia':
				docs = self.delete_trans(docs)	

			freq_sim_dict = self.Sim_cal.cal_sim_by_freq(docs)

			temp_freq_sim_dict = {}
			for (x,y), sim in freq_sim_dict.items():
				temp_freq_sim_dict[x+'--##--'+y] = sim

			self.sim_fw.write(json.dumps(temp_freq_sim_dict, ensure_ascii=False)+'\n')

			emb_sim_dict = self.Sim_cal.cal_sim_by_emb(docs)
			print( "the number of values", len(set(docs)))
		
		elif sims2 is not None:
			alpha = int(self.sim_type[-1]) *1.0 / 10
			print("alpha:", alpha)
			freq_sim_dict = self.Sim_cal.merge_sims(freq_sim_dict, sims2, alpha)
			temp_freq_sim_dict = {}
			for (x,y), sim in freq_sim_dict.items():
				temp_freq_sim_dict[x+'--##--'+y] = sim

			globle_fw.write(json.dumps(temp_freq_sim_dict, ensure_ascii=False)+'\n')
			
		#create graph
		G = nx.Graph()
		alpha = 1.0
		w_flag = False
		w_flag = True
		print('alpha:', alpha, 'w/o weighted:', w_flag)
		# print(freq_sim_dict)
		for (x, y), sim in freq_sim_dict.items():
			# if alpha != 1.0:
			# 	emb_sim = emb_sim_dict[(x,y)]
			# 	sim = sim*alpha+(1- alpha)*emb_sim
			if sim < self.thre:
				if x not in G:
					G.add_node(x)
				if y not in G:
					G.add_node(y)
				continue
			if w_flag:	
				G.add_edge(x, y, weight=sim) #) #,
			else:
				G.add_edge(x, y)
		return G

	def louvain(self, graph):
		idx = 0
		cluster = defaultdict(list)
		try:
			# partition = community.best_partition(graph, partition=None, weight='weight', resolution=1.1, randomize=None, random_state=1)
			partition = community.best_partition(graph, partition=None, weight='weight', resolution=0.8, randomize=None, random_state=1)
			# print("community.partition:", partition)
			mod = community.modularity(partition, graph)
			for node in graph.nodes():
				# print('node:', idx,  node, 'belongs to com:', partition[node])
				# idx += 1
				cluster[partition[node]].append(node)
		except:
			traceback.print_exc()
			print('best_partition error')
			idx = 0 
			for v in graph.nodes():
				cluster[idx].append(v)
				idx += 1
			# pass

		return cluster

	def get_clusters(self, docs=None, freq_sim_dict=None, sims2 = None):

		graph = self.create_graph(docs, freq_sim_dict, sims2)
		if self.com_type in ['louvain', None]:
			temp_res = self.louvain(graph)
		else:
			print('Unimplemented cluster methods')

		if isinstance(temp_res, dict):
			clusters = []
			for k, vs in temp_res.items():
				if dataset == 'dbpedia':
					vs = self.add_trans(vs)
				clusters.append(vs)
		else:
			temp_res = self.add_trans(temp_res)
			clusters = temp_res

		return clusters

class Cluster:
	def __init__(self, ci_flag=True):
		self.dataset = dataset
		ci_flag = False
		if self.dataset == 'dbpedia':
			if ci_flag:
				self.cleanres_file = self.dataset+'/stand_cleanres4' + self.dataset + '_ci_1.json'
			else:
				self.cleanres_file = self.dataset+'/stand_cleanres4' + self.dataset + '_ci_0.json'
		else:
			self.cleanres_file = self.dataset+'/stand_cleanres4' + self.dataset + '_ci.json'
				
		self.target_ps = None

		self.write_flag = True
		self.write_flag = False
		
		sim_type = 'text_emb_large_cp_w_f5'    #text_emb+cp_w_freq, 5:5
		sim_type = 'text_emb_large_wp_idf6'
		self.sim_type = sim_type

		print('sim_type: ', sim_type)
		suffix = '.json'
		# suffix = '_norm.json'

		self.wp_file = 'word_piece_file_'+sim_type+'_'+str(int(ci_flag))+suffix
		self.sim_file = self.dataset+'/sim_pairs_file_'+sim_type+'_'+str(int(ci_flag))+suffix
		self.tfidf_file = 'file_'+sim_type+'_'+str(int(ci_flag))+suffix

		print("target_ps:", self.target_ps)
		print('write_flag:', self.write_flag)
		print('wp_file:', self.wp_file)

		if self.write_flag:
			if 'cp' in sim_type or 'wp' in sim_type:
				self.wp_fw = open(self.wp_file, 'w', encoding='utf8')
				self.wp_fw.write('')

				self.idf_fw = open(self.tfidf_file, 'w', encoding='utf8')
				self.idf_fw.write('')
			else:
				self.wp_fw = None
				self.idf_fw = None

			self.sim_fw = open(self.sim_file, 'w', encoding='utf8')
			self.sim_fw.write('')

		else:
			self.wp_fw = None
			self.sim_fw = None
			self.idf_fw = None
		
		self.com_type = 'louvain'

		thre = 0.7
		thre = 0.0
		if 'louvain' in self.com_type:
			self.community_dis = Comunity_dis(sim_type=sim_type, com_type = self.com_type, wp_fw=self.wp_fw, sim_fw=self.sim_fw, idf_fw=self.idf_fw, thre=thre )
			self.cluster_res_file = self.dataset+'_cufen_res_'+sim_type+'_'+str(int(ci_flag))+'_'+self.com_type+str(thre)+suffix
		else:
			sim_type = ''
			self.sim_file = None
			if len(sim_type) != 0:
				self.sim_file = 'sim_pairs_file_'+sim_type+'_'+str(int(ci_flag))+suffix
			self.community_dis = Cluster_dis(clu_type = self.com_type)
			self.cluster_res_file = self.dataset+'/cluster_res/'+self.dataset+'_cufen_res_'+sim_type+'_'+self.com_type+suffix


	def __del__(self):
		if hasattr(self, 'wp_fw') and self.write_flag:
			if self.wp_fw is not None:
				self.wp_fw.close()
		if hasattr(self, 'sim_fw') and self.write_flag:
			self.sim_fw.close()
		if hasattr(self, 'idf_fw') and self.write_flag:
			if self.idf_fw is not None:
				self.idf_fw.close()

	def get_final_clusters(self):
		if ('dbscan' in self.com_type) or ('kmeans' in self.com_type):
			self.get_tra_clusters()
		else:
			self.get_com_clusters()

	def get_tra_clusters(self):
		with open(self.cleanres_file, 'r', encoding='utf8') as fr:
			data = json.load(fr)

		all_sim_pairs = {}
		if self.sim_file is not None:
			with open(self.sim_file, 'r', encoding='utf8') as fr:
				line = fr.readline()
				while line:
					if 'the attribute currently processed is' in line:
						attr = fr.readline().strip()
						line = fr.readline()
						con = fr.readline().strip()
						line = fr.readline()
						if attr not in all_sim_pairs:
							all_sim_pairs[attr] = {}
						all_sim_pairs[attr][con] = {}

						# print(attr)
						sim_pairs = json.loads(line)
						for k, sim in sim_pairs.items():
							(x, y) = k.split('--##--')
							all_sim_pairs[attr][con][(x,y)] = sim
					line = fr.readline()


		res = {}
		for attr, info in data.items():
			res[attr] = {}
			# if self.dataset == 'dbpedia':
			# 	info = {'all':info}
			for con, c_info in info.items():
				print('----'*10)
				print('the attribute and concept currently processed is:', attr, con)
				if all_sim_pairs:
					sim_pairs = all_sim_pairs[attr][con]
				else:
					sim_pairs = None
				# print(c_info.keys())
				clus_res = self.community_dis.get_clusters(list(c_info.keys()), sim_pairs, self.n_dict[attr][con])
				# print(clus_res)
				res[attr][con] = clus_res
			# 	break
			# break
		with open(self.cluster_res_file, 'w', encoding='utf8') as fw:
			json.dump(res, fw, ensure_ascii=False, indent=4) 

	def get_com_clusters(self):
		if self.write_flag:
			#相似度没计算好
			with open(self.cleanres_file, 'r', encoding='utf8') as fr:
				data = json.load(fr)

			# data = {'国籍':{'人物':{'中国杭州':[['中国杭州'], 1], '中国x':[['中国x'], 7], '中国温州':[['中国温州'], 1]}}}
			res = {}
			for attr, info in data.items():
				if self.target_ps:
					if attr not in self.target_ps:
						continue
				res[attr] = {}
				# if self.dataset == 'dbpedia':
				# 	info = {'all':info}

				for con, c_info in info.items():
					print('----'*10)
					self.sim_fw.write('--------the attribute currently processed is:\n' )
					self.sim_fw.write(attr+'\n')
					self.sim_fw.write('--------the concept currently processed is:\n' )
					self.sim_fw.write(con+'\n')
					if self.wp_fw:
						self.wp_fw.write('--------the attribute currently processed is:\n' )
						self.wp_fw.write(attr+'\n')
						self.wp_fw.write('--------the concept currently processed is:\n' )
						self.wp_fw.write(con+'\n')
						self.idf_fw.write('--------the attribute currently processed is:\n' )
						self.idf_fw.write(attr+'\n')
						self.idf_fw.write('--------the concept currently processed is:\n' )
						self.idf_fw.write(con+'\n')
					print('the attribute and concept currently processed is:', attr, con)
					# temp_info = {}
					# i = 0
					# for k, v in c_info.items():
					# 	temp_info[k] = v
					# 	i += 1
					# 	if i == 20:
					# 		break
					clusters = self.community_dis.get_clusters(c_info)
					res[attr][con] = clusters
				# 	break
				# break

		else:
			if os.path.exists(self.sim_file):
				res = {}
				with open(self.sim_file, 'r', encoding='utf8') as fr:
					line = fr.readline()
					while line:
						if 'the attribute currently processed is' in line:
							attr = fr.readline().strip()
							line = fr.readline()
							con = fr.readline().strip()
							line = fr.readline()
							
							if attr not in res:
								res[attr] = {}
							print(attr)
							sim_pairs = json.loads(line)
							freq_sim_dict = {}
							for k,sim in sim_pairs.items():
								(x, y) = k.split('--##--')
								freq_sim_dict[(x,y)] = sim
							clusters = self.community_dis.get_clusters(None, freq_sim_dict)	
							res[attr][con] = clusters
						line = fr.readline()

			elif 'text_emb_large_' in self.sim_type:  #组合式相似度计算
				sim_file1 = 'sim_pairs_file_text_emb_large_0.json'
				sims1 = {}
				res = {}
				with open(sim_file1, 'r', encoding='utf8') as fr:
					line = fr.readline()
					while line:
						if 'the attribute currently processed is' in line:
							attr = fr.readline().strip()
							line = fr.readline()
							con = fr.readline().strip()
							line = fr.readline()
							if attr not in res:
								res[attr] = {}
								sims1[attr] = {}
							sims1[attr][con] = {}

							# print(attr)
							sim_pairs = json.loads(line)
							for k, sim in sim_pairs.items():
								(x, y) = k.split('--##--')
								sims1[attr][con][(x,y)] = sim
						line = fr.readline()
							
				if 'cp_wo_f' in self.sim_type:
					sim_file2 = 'sim_pairs_file_cp_w_freq_0.json'
				elif 'cp_w_f' in self.sim_type:
					sim_file2 = 'sim_pairs_file_cp_wo_freq_0.json'
				elif 'tfidf' in self.sim_type:
					sim_file2 = 'sim_pairs_file_wp_tfidf_0.json'

				sims2 = {}
				with open(sim_file2, 'r', encoding='utf8') as fr:
					line = fr.readline()
					while line:
						if 'the attribute currently processed is' in line:
							attr = fr.readline().strip()
							line = fr.readline()
							con = fr.readline().strip()
							line = fr.readline()
							if attr not in sims2:
								sims2[attr] = {}
							sims2[attr][con] = {}

							# print(attr)
							sim_pairs = json.loads(line)
							for k, sim in sim_pairs.items():
								(x, y) = k.split('--##--')
								sims2[attr][con][(x,y)] = sim
						line = fr.readline()

				for attr, info in sims1.items():
					for con, c_info in info.items():
						globle_fw.write('--------the attribute currently processed is:\n' )
						globle_fw.write(attr+'\n')
						globle_fw.write('--------the concept currently processed is:\n' )
						globle_fw.write(con+'\n')
						temp_sims2 = sims2[attr][con]
						clusters = self.community_dis.get_clusters(None, c_info, temp_sims2)
						res[attr][con] = clusters
			else:
				print('unconsidered cases,', self.sim_type)
				exit()
				

		globle_fw.close()				
		with open(self.cluster_res_file, 'w', encoding='utf8') as fw:
			json.dump(res, fw, ensure_ascii=False, indent=4)

	
if __name__ == '__main__':
	clu = Cluster()
	clu.get_final_clusters()
