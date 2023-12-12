import json


def prepare_data(path, threshold=None):
    questions, answers = [], []
    if "WQ" in path:
        with open(path, 'r', encoding='utf8') as fp:
            json_data = json.load(fp)
            items = json_data['Questions']
            for item in items:
                question = item['ProcessedQuestion']
                # print(item['Parses'][0]['Answers'])
                answer = []
                for a in item['Parses'][0]['Answers']:
                    if a['AnswerType'] == 'Entity':
                        answer.append(a['EntityName'].lower())
                    elif a['AnswerType'] == 'Value':
                        answer.append(a['AnswerArgument'])
                answer = list(set(answer))
                answer = ','.join(answer)
                if threshold is None or (len(answer) <= threshold and len(question) <= threshold):
                    questions.append(question)
                    answers.append(answer)
    return questions, answers


def format_prompt(desc, demo, question):
    prompt = desc + "\n" + demo + "\n" + question
    return prompt


def format_user_chat(demo, question):
    prompt = demo + "\n" + question
    return prompt


def combine_cache(path1, path2):
    import pickle
    f = open(path1, 'rb')
    cache1 = pickle.load(f)
    f = open(path2, 'rb')
    cache2 = pickle.load(f)
    cache1.update(cache2)
    f = open(path1, 'wb')
    pickle.dump(cache1, f)
    f.close()
