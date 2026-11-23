import datetime

from .config import CATEGORIES

CATEGORY_ANCHORS = '''大类锚点示例（辅助 category 判断）：
- 人力资源：请（休）假、考勤、薪酬、补贴津贴（住房补贴、岗位津贴）、企业年金、集体户口、劳务人员转正、返聘退休
- 财务管理：差旅费、借款报销、预算、票据、资金、会计岗位
- 党的建设：党支部、新闻宣传、工会、共青团、纪委、党风廉政
- 公共管理：印章证照、车辆、会议、接待、安全保卫、宿舍、人才租赁住房、保密
- 科技创新：知识产权、核心技术、资质认证
- 科研生产：科技成果、科技奖励（论文/成果奖励）、科研生产项目、外场试验
- 考核办法：各类专项考核细则
- 法人治理：章程、董事会、股东会、三重一大
- 内部监督/市场营销/质量技术/资产经营：内部控制评价、市场考核、质量、固定资产'''

SYSTEM_TEMPLATE = '''你是企业制度智能问答系统的查询理解模块。你的任务是把员工的口语化提问改写成结构化查询，供后续检索公司制度文档使用。

【硬性要求】
1. 只输出一个 JSON 对象，不要输出任何其他文字、解释或 markdown 代码块。
2. 忠实原意：不得引入用户没有提到的事项、主体或数字；不得回答用户的问题本身。
3. 术语对齐：keywords 和 sub_queries 必须使用公文规范措辞（参考下方术语表），不要用口语词；涉及具体制度时，必须使用下方制度全称清单中的全称——用户提到简写或近似名时校正为全称（如用户说 预算管理制度，应写 全面预算管理办法）。
4. 今天是 {today}。涉及制度版本时按 time_scope 标注。

{anchors}

【制度全称清单（按大类）】
{titles}

【输出 JSON 字段】
- standalone_query：结合对话历史消解指代后、可独立理解的完整问题（中文陈述句）。
- intent：只能是以下之一——事实数字 / 流程手续 / 条件资格 / 计算判断 / 制度元信息 / 闲聊超域。
- keywords：3~6 个公文术语关键词，用于关键词检索（如 带薪年休假、差旅费、出差申请单）。
- sub_queries：2~3 个不同角度的公文问法改写（与 standalone_query 不重复，用于语义检索）。
- category：问题所属制度大类，只能是：{categories}。闲聊或无法判断时填空字符串。
- subject：问题针对的主体（如 员工、劳务人员、女职工、领导人员、返聘人员），无则空字符串。
- time_scope：latest（问最新/现行版本）/ specific（问特定旧版本）/ unspecified。
- need_retrieval：是否需要检索制度库。闲聊、打招呼、制度外问题填 false，intent 同时为闲聊超域。

【领域术语表（公文术语（口语说法））】
{glossary}
'''

FEWSHOT = [
    (
        '老员工休假怎么算？', None,
        {
            'standalone_query': '老员工的带薪年休假天数如何根据累计工作年限计算',
            'intent': '事实数字',
            'keywords': ['带薪年休假', '累计工作年限', '年休假天数'],
            'sub_queries': ['员工累计工作年限对应的带薪年休假天数规定', '请（休）假管理办法中年休假计算规则'],
            'category': '人力资源', 'subject': '员工', 'time_scope': 'unspecified', 'need_retrieval': True,
        },
    ),
    (
        '出差坐飞机报销要什么手续？', None,
        {
            'standalone_query': '员工出差乘坐飞机的差旅费报销需要哪些手续和审批',
            'intent': '流程手续',
            'keywords': ['差旅费', '城市间交通费', '出差申请单', '报销审批'],
            'sub_queries': ['出差乘坐飞机的审批流程和权限规定', '差旅费报销所需单据与程序'],
            'category': '财务管理', 'subject': '员工', 'time_scope': 'unspecified', 'need_retrieval': True,
        },
    ),
    (
        '那病假呢？', ['婚假能休几天？', '根据《请（休）假管理办法》，婚假共10天，自结婚登记之日起一年内有效。'],
        {
            'standalone_query': '员工请病假的天数和工资待遇如何规定',
            'intent': '事实数字',
            'keywords': ['病假', '医疗期', '病假工资'],
            'sub_queries': ['请（休）假管理办法中病假天数与医疗期规定', '病假期间工资待遇标准'],
            'category': '人力资源', 'subject': '员工', 'time_scope': 'unspecified', 'need_retrieval': True,
        },
    ),
    (
        '食堂今天中午吃什么？', None,
        {
            'standalone_query': '食堂今天中午吃什么',
            'intent': '闲聊超域',
            'keywords': [], 'sub_queries': [],
            'category': '', 'subject': '', 'time_scope': 'unspecified', 'need_retrieval': False,
        },
    ),
    (
        '现在执行的全面预算管理办法是哪一版？', None,
        {
            'standalone_query': '公司现行有效的全面预算管理办法版本',
            'intent': '制度元信息',
            'keywords': ['全面预算管理办法', '版本', '文号'],
            'sub_queries': ['全面预算管理办法最新修订版本', '全面预算管理制度现行文号'],
            'category': '财务管理', 'subject': '', 'time_scope': 'latest', 'need_retrieval': True,
        },
    ),
]


def build_messages(query, history, glossary_digest, categories=None, titles=''):
    import json
    cats = '、'.join(categories or CATEGORIES)
    system = SYSTEM_TEMPLATE.format(
        today=datetime.date.today().isoformat(),
        categories=cats,
        glossary=glossary_digest,
        anchors=CATEGORY_ANCHORS,
        titles=titles,
    )
    messages = [{'role': 'system', 'content': system}]
    for q, h, ans in FEWSHOT:
        user_text = q if not h else ('对话历史：' + ' / '.join(h) + chr(10) + '当前问题：' + q)
        messages.append({'role': 'user', 'content': user_text})
        messages.append({'role': 'assistant', 'content': json.dumps(ans, ensure_ascii=False)})
    if history:
        hist = [('用户：' if i % 2 == 0 else '助手：') + t for i, t in enumerate(history)]
        user_text = '对话历史：' + ' / '.join(hist) + chr(10) + '当前问题：' + query
    else:
        user_text = query
    messages.append({'role': 'user', 'content': user_text})
    return messages
