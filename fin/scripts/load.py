import re


# 객관식 여부 판단 함수
def is_multiple_choice(question_text):
    """
    객관식 여부를 판단: 2개 이상의 숫자 선택지가 줄 단위로 존재할 경우 객관식으로 간주
    """
    lines = question_text.strip().split("\n")
    option_count = sum(bool(re.match(r"^\s*[1-9][0-9]?\s", line)) for line in lines)
    return option_count >= 2


# 질문과 선택지 분리 함수
def extract_question_and_choices(full_text):
    """
    전체 질문 문자열에서 질문 본문과 선택지 리스트를 분리
    """
    lines = full_text.strip().split("\n")
    q_lines = []
    options = []

    for line in lines:
        if re.match(r"^\s*[1-9][0-9]?\s", line):
            options.append(line.strip())
        else:
            q_lines.append(line.strip())
    
    question = " ".join(q_lines)
    return question, options


# # 프롬프트 생성기
# def make_prompt_auto(text):
#     if is_multiple_choice(text):
#         question, options = extract_question_and_choices(text)
#         prompt = (
#                     "당신은 금융보안 전문가입니다.\n"
#                     "아래 질문에 대해 적절한 **정답 선택지 번호만 출력**하세요.모든 답변은 한국어를 사용하며, 다른 언어나 특수기호는 포함하지 마시오.\n\n"
#                     f"질문: {question}\n"
#                     "선택지:\n"
#                     f"{chr(10).join(options)}\n\n"
#                     "답변:"
#                 )
#     else:
#         prompt = (
#                     "당신은 금융보안 전문가입니다.\n"
#                     "아래 주관식 질문에 대해 정확하고 간략한 설명을 작성하세요.모든 답변은 한국어를 사용하며, 다른 언어나 특수기호는 포함하지 마시오.\n\n"
#                     f"질문: {text}\n\n"
#                     "답변:"
#                 )   
#     return prompt

# 프롬프트 생성기 (Chat 형식)
def make_prompt_auto(text):
    if is_multiple_choice(text):
        question, options = extract_question_and_choices(text)
        return [
            {"role": "system", "content": "당신은 금융보안 전문가입니다. 모든 답변은 **한국어**만을 사용하며, 다른 언어나 특수기호는 포함하지 마세요. 한자를 출력하지 마세요.\n\n"},
            {"role": "user", "content":
                "아래 질문에 대해 적절한 **정답 선택지 번호를 하나만 출력**하세요.\n"
                f"질문: {question}\n"
                "선택지:\n"
                f"{chr(10).join(options)}\n\n"
                "답변:"
            }
        ]
    else:
        return [
            {"role": "system", "content": "당신은 금융보안 전문가입니다. 모든 답변은 **한국어**만을 사용하며, 다른 언어나 특수기호는 포함하지 마세요. 한자를 출력하지 마세요.\n\n"},
            {"role": "user", "content":
                "아래 주관식 질문에 대해 정확하고 간략한 답변을 작성하세요.\n"
                f"질문: {text}\n\n"
                "답변:"
            }
        ]