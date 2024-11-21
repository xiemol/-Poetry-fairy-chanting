import re
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from chatmodel import openai


#学习完所有纵横术进行考核

def extract_first_number_advanced(text):
    """使用更高级的正则表达式，可以处理负数和小数"""
    match = re.search(r'-?\d+(?:\.\d+)?', text)
    return float(match.group()) if match else None


def game2():
    prompt_str = """
    请按照以下格式提供输出:
    分数：   ；原因：(只需一句简短的文字评判我的选择即可)

        你是一位中国古代的纵横家教师，请你按照我给你的判分标准判分，以下是我的题目详情和判分标准
    \n\n题目详情:
        请根据以下情境,选择您认为最合适的策略。注意,每个选项都有其优缺点,没有绝对的对错之分。
    您是一个小国的君主,面对两个强大邻国的威胁。您会选择:
    一. 与其中一个强国结盟,对抗另一个
    二. 试图与两个强国都保持友好关系
    三. 联合其他小国,形成联盟以对抗强国
    四. 专注于内政发展,提升自身实力
    在与敌对国家的谈判中,您发现对方内部存在矛盾。您会:
    一. 利用反间计,加剧对方内部矛盾
    二. 寻找共同利益,促进和平谈判
    三. 装作不知情,专注于自身诉求
    四. 公开这一信息,试图在谈判中占据主动
    您的盟友正在与您的潜在敌人交好。您会:
    一. 威胁断绝与盟友的关系
    二. 寻找新的盟友来平衡局势
    三. 主动与潜在敌人改善关系
    四. 增加对盟友的利益输送,巩固关系
    在一场多国会谈中,您需要推动一项对您有利但可能引起争议的提案。您会:
    一. 直接强硬推动,展示决心
    二. 私下游说,寻求关键国家的支持
    三. 提出一个更激进的方案,然后退而求其次
    四. 将提案包装成对多数国家都有利的形式
    您的国家在技术上落后于邻国,但农业资源丰富。您会:
    一. 全力发展农业,成为地区粮仓
    二. 通过贸易换取技术,谋求平衡发展
    三. 派间谍窃取先进技术
    四. 联合其他技术落后国家,抵制先进国家
    \n\n判分标准:
    本测试旨在评估考生对纵横术核心理念的理解和应用能力。评分应基于以下几个方面:

    策略的一致性 (25分)

    考生的选择是否体现出一致的战略思维?
    例如:选择1.三、2.一、3.二、4.二、5.二可能表明考生倾向于联盟和平衡策略。


    情况判断能力 (25分)

    考生是否根据题目中的具体情况做出恰当判断?
    例如:在问题2中选择一可能显示出对形势的敏锐把握。


    灵活性和创新性 (20分)

    考生的选择是否展现出灵活应变的能力?
    例如:在问题4中选择三展示了谈判中的策略性思维。


    长远考虑 (20分)

    考生的选择是否考虑到长期利益和可持续发展?
    例如:在问题5中选择二体现了长远发展的思考。


    道德和外交平衡 (10分)

    考生是否在追求利益的同时也考虑到道德和外交影响?
    例如:在问题3中选择四而不是一显示出对外交关系的重视。



    总分: 100分
    注意:这个评分标准强调的是考生思维的连贯性和对复杂情况的理解,而不是简单地判断对错。评分者应该根据考生的整体表现来评判,而不是机械地根据单个答案打分。
    \n
    以下是我每一题对应的选择：\n
        {text}\n
    请你联系上述题目和评分标准给我评一个分。
    """

    prompt = PromptTemplate(
        input_variables=["text"],
        template=prompt_str,
    )
    chain = LLMChain(
        prompt=prompt,
        llm=openai(),
    )
    return chain

