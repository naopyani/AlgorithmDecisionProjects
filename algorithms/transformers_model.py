from transformers import AutoTokenizer, pipeline, AutoModelForSequenceClassification


def init_pipeline_classifier(path):
    # 加载本地模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    return pipeline("text-classification", model=model, tokenizer=tokenizer)


def identify_user_intent(text):
    """
    加载不同任务的模型来识别用户意图
    :param text: 意图文本
    :return:
    """
    # 识别用户是否想要查看数据分布
    distribution_classifier = init_pipeline_classifier("./models/distribution_model")
    # 识别用户是否想要数据线性相关情况
    relation_classifier = init_pipeline_classifier("./models/relation_model")
    # 识别用户是否想要做模型训练
    train_classifier = init_pipeline_classifier("./models/train_model")
    result_list = [distribution_classifier(text)[0]["label"], relation_classifier(text)[0]["label"],
                   train_classifier(text)[0]["label"]]
    if "LABEL_1" in result_list:
        return result_list.index(max(result_list))
    raise RuntimeError("对不起，我识别不出您的意图")


if __name__ == '__main__':
    problem_number = identify_user_intent("查看特征分布情况")
    print(problem_number)
