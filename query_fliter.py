import argparse
import json
import jsonlines

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_path")
    parser.add_argument("--gpt_path")
    args = parser.parse_args()

    already_users = set()
    gpt_answers_finished = []

    with open(args.gpt_path, "r") as f:
        for i, item in enumerate(jsonlines.Reader(f)):
            user = item[0]['user']
            answers = []
            try:
                if 'chat' not in item[1]['object']:
                    for choice in item[1]['choices']:
                        if choice['finish_reason'] == 'stop':
                            answers.append(choice['text'])
                        else:
                            answers.append("")
                else:
                    for choice in item[1]['choices']:
                        if choice['finish_reason'] == 'stop':
                            answers.append(choice['message']['content'])
                        else:
                            answers.append("")
                gpt_answers_finished.append(item)
                already_users.add(user)
            except:
                print(user)

    new_query = []
    with open(args.query_path, "r+", encoding="utf-8") as f:
        for i, item in enumerate(jsonlines.Reader(f)):
            if item['user'] not in already_users:
                new_query.append(item)

    query_output_path = args.query_path.replace(".json", "_filter.json")
    with open(query_output_path, "w") as f:
        for json_query in new_query:
            json_string = json.dumps(json_query)
            f.write(json_string + "\n")

    gpt_output_path = args.gpt_path.replace(".json", "_filter.json")
    with open(gpt_output_path, "w") as f:
        for json_query in gpt_answers_finished:
            json_string = json.dumps(json_query)
            f.write(json_string + "\n")
