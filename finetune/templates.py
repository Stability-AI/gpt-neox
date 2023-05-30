# prompt templates
INPUT_PROMPT = """以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。

### 指示: 
{instruction}

### 入力: 
{input}

### 応答: 
{response}"""

NO_INPUT_PROMPT = """以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。

### 指示: 
{instruction}

### 応答: 
{response}"""