# Prompt-Engineering
113-2 電機所 生成式AI HW1 Prompt Engineering

## Author：國立陽明交通大學 資訊管理與財務金融學系財務金融所碩一 313707043 翁智宏

本次是生成式AI課程的第一次作業，是做 Prompt Engineering，主要利用公開且免費閉源的LLM API ( e.g. Gemini、Groq、DeepSeek )，去調整prompting，來提高回答問題的準確率。

**Questions**
Try as many prompt engineering techniques as you can.

## Dataset Description
- There are some information you can use in **mmlu_sample.csv**
- Generate your answer based on the question in **mmlu_submit.csv**
- You need to follow the format in **submit_format.csv**
- You need to update the ID and the answer in submit_format.csv that matched the ID of the question in mmlu_submit.csv

**mmlu_sample.csv**
---
會有 **"input"、"A"、"B"、"C"、"D"、"target"、"task"** 欄位，
分別代表，**"題目"、"選項"、"答案"、"題目種類"**。

範例：

**input**：
This question refers to the following information.
The following quote is from Voltaire in response to the 1755 Lisbon earthquake.
My dear sir, nature is very cruel. One would find it hard to imagine how the laws of movement cause such frightful disasters in the best of possible worlds. A hundred thousand ants, our fellows, crushed all at once in our ant-hill, and half of them perishing, no doubt in unspeakable agony, beneath the wreckage from which they cannot be drawn. Families ruined all over Europe, the fortune of a hundred businessmen, your compatriots, swallowed up in the ruins of Lisbon. What a wretched gamble is the game of human life! What will the preachers say, especially if the palace of the Inquisition is still standing? I flatter myself that at least the reverend father inquisitors have been crushed like others. That ought to teach men not to persecute each other, for while a few holy scoundrels burn a few fanatics, the earth swallows up one and all.
?oltaire, in a letter, 1755
The ideas expressed by Voltaire, above, best illustrate which of the following characteristics of Enlightenment intellectuals?

**A：**　Many were accomplished scientists, who added important pieces to human understanding of the universe.	

**B：**　They utilized new methods of communicating their ideas, such as salons and inexpensive printed pamphlets.	

**C：**　Most rejected religion altogether and adopted atheism as the only credo of a rational man.	

**D：**　Many believed that the new scientific discoveries justified a more tolerant and objective approach to social and cultural issues.	

**target：** D

**task：** Dhigh_school_european_history		

**mmlu_submit.csv**
---
會有 **"input"、"A"、"B"、"C"、"D"、"task"** 欄位，與範例缺少了 **target**。

**kaggle Competition**
---
競賽連結：[點擊這裡](https://www.kaggle.com/competitions/hw-1-prompt-engineering/overview) 

![最終成績](最終成績.png)

> Public 準確率在 0.96108 ，Private 準確率在 0.95785 (越高越精確)

 ![最終排名](最終排名.png)
> 全班共約有130人，private排名13名
 ![最終排名](最終成績_public.png)
> public排名30名

## 模型介紹
使用了 **Gemini-2.0-flash** 和 **Gemini-1.5-pro**，利用 **LangChain**。

## 安裝依賴
請使用以下指令安裝本專案所需的依賴套件：
```bash
pip install -U google-generativeai
pip install google-generativeai langchain langchain-google-genai
```
---

## 實作 (這邊介紹 Gemini-2.0-flash 為例)
### 第一步：設計 few-shot 範例格式，挑五筆 sample ( 2 個相似範例 + 3 個隨機範例 )
1) 定義相似度題型
```bash
def extract_answer(response):
    patterns = [
        r"Correct answer:\s*([A-D])",
        r"Answer:\s*([A-D])",
        r"The correct option is\s*([A-D])",
        r"I choose\s*([A-D])",
        r"Final answer:\s*([A-D])"
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    lines = response.strip().split('\n')
    last_line = lines[-1].strip()
    match = re.search(r"[A-D]", last_line)
    if match:
        return match.group(0).upper()
    return random.choice(["A", "B", "C", "D"])
```

2) 動態挑選題型，使用 Sentence-BERT ，根據上下文關聯找相似題型
```bash
# 2 個相似範例 + 3 個隨機範例
    similar_examples = get_similar_examples(row["input"], task_examples, 3)
    random_examples = task_examples.sample(min(2, len(task_examples)), random_state=42)
    few_shot_examples = pd.concat([similar_examples, random_examples]).drop_duplicates()
```

3) 定義任務、Role's Prmpt
```bash
task_strategy = category_strategies.get(
        task,
        "Implement a comprehensive strategy that leverages detailed domain expertise and rigorous logical reasoning."
    )
role = role_dict.get(task, "a seasoned expert with extensive domain-specific knowledge")
```

4) 定義 few-shot 範例格式
```bash
few_shot_text = "\n".join([
        f"""Example {i+1}:
            Question: {ex['input']}
            A) {ex['A']}
            B) {ex['B']}
            C) {ex['C']}
            D) {ex['D']}
            Correct Answer: {ex['target']}"""
for i, (_, ex) in enumerate(few_shot_examples.iterrows())
])
```

### 第二步：設計 Prompt Engineering ( 使用 CoT )
1) 定義 **task** 種類
```bash
category_strategies = {
    "high_school_biology": "Delve into fundamental biological concepts, technical terminology, and biological processes.",
    "high_school_computer_science": "Analyze programming logic, algorithms, and principles of software design.",
    "high_school_european_history": "Examine European historical contexts, key events, and their cause-effect relationships.",
    "high_school_geography": "Focus on spatial distributions, natural landforms, and human-environment interactions.",
    "high_school_government_and_politics": "Evaluate political systems, governmental operations, and core political theories.",
    "high_school_macroeconomics": "Explore macroeconomic principles, market dynamics, and the impact of policies.",
    "high_school_microeconomics": "Concentrate on individual markets, supply-demand interactions, and consumer behavior.",
    "high_school_psychology": "Study psychological theories, behavioral patterns, and cognitive processes.",
    "high_school_us_history": "Review significant events, figures, and developmental trends in U.S. history.",
    "high_school_world_history": "Assess global historical trends, cultural exchanges, and international influences."
}
```

2) 定義 **role** 種類
```bash
role_dict = {
    "high_school_biology": "a biology professor specializing in high school curricula",
    "high_school_computer_science": "a computer science professor with expertise in programming logic",
    "high_school_european_history": "a European history expert focused on high school education",
    "high_school_geography": "a geography educator specializing in spatial analysis",
    "high_school_government_and_politics": "a political science scholar with knowledge of governmental systems",
    "high_school_macroeconomics": "a macroeconomics professor specializing in economic policy",
    "high_school_microeconomics": "a microeconomics professor focused on market dynamics",
    "high_school_psychology": "a psychology instructor with expertise in behavioral theories",
    "high_school_us_history": "a U.S. history expert specializing in key events and trends",
    "high_school_world_history": "a world history specialist focused on global trends"
}
```
3) CoT
```bash
prompt_template = PromptTemplate(
    template="""
        You are {role} specializing in solving multiple-choice questions with high accuracy. The current question is from {task}.

            🔹 **Rules**:
            1. Output only the final answer in this exact format: 'Correct answer: X' (where X is A, B, C, or D).
            2. Include only the output, avoiding reasoning or additional text.
            
            🔹 **Instructions**:
            Solve the question by reasoning step-by-step:
            1. Identify the core concept or fact the question is testing.
            2. Analyze each option:
               - Assess its alignment with the core concept.
               - Note why it might be correct or incorrect.
            3. Eliminate incorrect options:
               - Identify flaws (e.g., factual errors, misinterpretations).
               - Watch for traps (e.g., subtle wording differences).
            4. Confirm the answer:
               - Ensure the remaining option fully answers the question.
            
            🔹 **Strategy**:
            Use these strategies for {task} questions: {task_strategy}.
            
            📌 **Examples**:
            {few_shot}
            
            Now solve this question:
            
            Question: {question}
            A) {A}
            B) {B}
            C) {C}
            D) {D}
""",
input_variables=["role", "task", "task_strategy", "few_shot", "question", "A", "B", "C", "D"]
)
```
   
