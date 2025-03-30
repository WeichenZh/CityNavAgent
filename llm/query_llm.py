import json
import os
import time
import re
import sys

# import openai                       # for OpenAI_LLM_v2
from openai import OpenAI           # for OpenAI_LLM_v3
from openai import AzureOpenAI
import base64

import unicodedata

# import dashscope
# from dashscope import MultiModalConversation
from http import HTTPStatus

# sys.path.append("../..")
# from groundingdino.util.inference import load_model, load_image, predict, annotate, annotate_predict

class LLM:
    def __init__(self, api_key, model_name, max_tokens, cache_name='default', **kwargs):
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.queried_tokens = 0

        cache_model_dir = os.path.join('llm', 'cache', self.model_name)
        os.makedirs(cache_model_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_model_dir, f'{cache_name}.json')
        self.cache = dict()

        if os.path.isfile(self.cache_file):
            with open(self.cache_file) as f:
                self.cache = json.load(f)

    def query_api(self, prompt):
        raise NotImplementedError

    def get_cache(self, prompt, instance_idx):
        sequences = self.cache.get(instance_idx, [])

        for sequence in sequences:
            if sequence.startswith(prompt) and len(sequence) > len(prompt)+1:
                return sequence
        return None

    def add_to_cache(self, sequence, instance_idx):
        if instance_idx not in self.cache:
            self.cache[instance_idx] = []
        sequences = self.cache[instance_idx]

        # newest result to the front
        sequences.append(sequence)

    def save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)
        print('cache saved to: ' + self.cache_file)

    def get_sequence(self, prompt, instance_idx, read_cache=True):
        sequence = None
        if read_cache:
            sequence = self.get_cache(prompt, instance_idx)
        print('cached sequence')
        if sequence is None:
            print('query API')
            sequence = self.query_api(prompt)
            self.add_to_cache(sequence, instance_idx)
            #print('api sequence')
        return sequence

class Qwen_VL(LLM):
    def __init__(self, model_name, api_key, logit_bias=None, max_tokens=64, finish_reasons=None, **kwargs):
        dashscope.api_key = api_key
        self.logit_bias = logit_bias

        super().__init__(api_key, model_name, max_tokens, **kwargs)

    def clean_grounding(self, response):
        cleaned_captions = []
        captions = response['output']['choices'][0]['message']['content']
        for i in range(len(captions)):
            if "box" in captions[i]:
                pattern = r'<ref>(.*?)</ref>.*?((?:<box>.*?</box>)*)(?:<quad>.*?</quad>)*'
                matches = re.findall(pattern, captions[i]["box"])
                for o, bbs in matches:
                    object = o.strip()
                    bboxes = []
                    bb_matches = re.findall(r'<box>(.*?)</box>', bbs)
                    for bb in bb_matches:
                        pps = re.findall(r'\d+', bb)
                        bboxes.append([(float(pps[0]), float(pps[1])), (float(pps[2]), float(pps[3]))])

                    cleaned_captions.append({"object": object, "bboxes": bboxes})

        return cleaned_captions

    def query_api(self, prompt, image_path=None, show_response=True):
        if image_path:
            image_path = 'file://{}'.format(image_path)  # 将图片路径转化为prompt标准格式
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"image": image_path},
                        {"text": prompt}
                    ]
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"text": prompt}
                    ]
                }
            ]
        response = MultiModalConversation.call(model=self.model_name, messages=messages)
        if response.status_code == HTTPStatus.OK:
            cleaned_grounding = self.clean_grounding(response)
            return cleaned_grounding, response['output']['choices'][0]['message']['content']
        else:
            print()
            print(response.code)  # The error code.
            print(response.message)  # The error message.
            return response, None

class OpenAI_LLM(LLM):

    def __init__(self, model_name, api_key, logit_bias=None, max_tokens=64, finish_reasons=None, **kwargs):
        openai.api_key = api_key
        self.logit_bias = logit_bias

        self.finish_reasons = finish_reasons
        if finish_reasons is None:
            self.finish_reasons = ['stop', 'length']

        super().__init__(api_key, model_name, max_tokens, **kwargs)

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def query_apis(self, prompt, image_paths=[], show_response=True):
        def query_func():
            return openai.Completion.create(engine=self.model_name,
                                            prompt=prompt,
                                            temperature=0,
                                            max_tokens=self.max_tokens
                                            )

        if self.logit_bias:
            def query_func():
                return openai.Completion.create(engine=self.model_name,
                                                prompt=prompt,
                                                temperature=0,
                                                max_tokens=self.max_tokens,
                                                logit_bias=self.logit_bias
                                                )

        if 'gpt-4' in self.model_name:
            query_messages = {
                "role": "user", "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }

            if len(image_paths) > 0:
                # import numpy as np
                # import cv2
                for img_p in image_paths:

                    # img = np.zeros((480, len(image_paths)*660, 3), dtype=np.uint8)
                    #
                    # for i in range(len(image_paths)):
                    #     img_t = cv2.imread(image_paths[i])
                    #     img[:, 660*i:660*i+640, :] = img_t
                    #
                    # img_p = "/home/vincent/py-pro/AirVLN-main/AirVLN/files/rgb_obs.png"
                    # cv2.imwrite(img_p, img)
                    base64_image = self.encode_image(img_p)
                    query_messages["content"].append({
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{base64_image}"
                        })


            # print(query_messages)
            def query_func():
                completion = openai.ChatCompletion.create(model=self.model_name,
                                                          messages=[query_messages],
                                                          max_tokens=self.max_tokens,
                                                          temperature=0,
                                                          )
                text = completion['choices'][0]['message']['content']
                if text[0] != ' ':
                    text = ' ' + text
                completion['choices'][0]['text'] = text
                return completion

        try:
            response = query_func()
        except (openai.error.ServiceUnavailableError,
                openai.error.RateLimitError,
                openai.error.ServiceUnavailableError,
                openai.error.APIConnectionError) as e:
            print(e)
            self.save_cache()
            time.sleep(10)
            print('try again')
            return self.query_api(prompt)

        if show_response:
            print('API Response:')
            print(response)
            print('')
        self.queried_tokens += response['usage']['total_tokens']
        # sequence = prompt + response['choices'][0]['text']
        sequence = response['choices'][0]['text']
        assert response['choices'][0]['finish_reason'] in self.finish_reasons + [None]
        return sequence


    def query_api(self, prompt, image_path=None,system=None, show_response=True):

        def query_func():
            return openai.Completion.create(engine=self.model_name,
                                            prompt=prompt,
                                            temperature=0,
                                            max_tokens=self.max_tokens
                                            )

        if self.logit_bias:
            def query_func():
                return openai.Completion.create(engine=self.model_name,
                                                prompt=prompt,
                                                temperature=0,
                                                max_tokens=self.max_tokens,
                                                logit_bias=self.logit_bias
                                                )

        if 'gpt-4' in self.model_name:
            query_messages = {
                "role": "user", "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }

            if image_path is not None:
                base64_image = self.encode_image(image_path)
                query_messages["content"].append({
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
                })

            # print(query_messages)
            def query_func():
                completion = openai.ChatCompletion.create(model=self.model_name,
                                                          messages=[query_messages],
                                                          max_tokens=self.max_tokens,
                                                          temperature=0,
                                                          )
                text = completion['choices'][0]['message']['content']
                if text[0] != ' ':
                    text = ' ' + text
                completion['choices'][0]['text'] = text
                return completion

        try:
            response = query_func()
        # except (openai.APIError,
        #         openai.RateLimitError,
        #         openai.APIConnectionError) as e:
        except Exception as e:
            print(e)
            self.save_cache()
            time.sleep(10)
            print('try again')
            return self.query_api(prompt)

        if show_response:
            print('API Response:')
            print(response)
            print('')
        self.queried_tokens += response['usage']['total_tokens']
        # sequence = prompt + response['choices'][0]['text']
        sequence = response['choices'][0]['text']
        assert response['choices'][0]['finish_reason'] in self.finish_reasons + [None]
        return sequence

    def query_api_map_gpt(self, prompt, system=None, image_path=None, show_response=False):
        def query_func():
            return openai.Completion.create(engine=self.model_name,
                                            prompt=prompt,
                                            temperature=0,
                                            max_tokens=self.max_tokens
                                            )

        if self.logit_bias:
            def query_func():
                return openai.Completion.create(engine=self.model_name,
                                                prompt=prompt,
                                                temperature=0,
                                                max_tokens=self.max_tokens,
                                                logit_bias=self.logit_bias
                                                )

        if 'gpt-4' in self.model_name:
            query_messages = [
                {
                    "role": "system",
                    "content": system
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            # if image_path is not None:
            #     base64_image = self.encode_image(image_path)
            #     query_messages["content"].append({
            #         "type": "image_url",
            #         "image_url": f"data:image/jpeg;base64,{base64_image}"
            #     })

            # print(query_messages)
            def query_func():
                completion = openai.ChatCompletion.create(model=self.model_name,
                                                          messages=query_messages,
                                                          max_tokens=self.max_tokens,
                                                          temperature=0,
                                                          )
                text = completion['choices'][0]['message']['content']
                if text[0] != ' ':
                    text = ' ' + text
                completion['choices'][0]['text'] = text
                return completion

        try:
            response = query_func()
        except (openai.error.APIStatusError,
                openai.error.RateLimitError,
                openai.error.APIConnectionError) as e:
            print(e)
            self.save_cache()
            time.sleep(10)
            print('try again')
            return self.query_api(prompt)

        if show_response:
            print('API Response:')
            print(response)
            print('')
        self.queried_tokens += response['usage']['total_tokens']
        # sequence = prompt + response['choices'][0]['text']
        sequence = response['choices'][0]['text']
        assert response['choices'][0]['finish_reason'] in self.finish_reasons + [None]
        return sequence

# if you need to use OpenAI_LLM_v2, degrade openai to version: pip install openai==0.27.8
class OpenAI_LLM_v2(LLM):
    def __init__(self, model_name, api_key, logit_bias=None, max_tokens=64, finish_reasons=None, **kwargs):
        openai.api_key = api_key
        self.logit_bias = logit_bias

        self.finish_reasons = finish_reasons
        if finish_reasons is None:
            self.finish_reasons = ['stop', 'length']

        super().__init__(api_key, model_name, max_tokens, **kwargs)

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def query_apis(self, prompt, image_paths=[], show_response=True):
        def query_func():
            return openai.Completion.create(engine=self.model_name,
                                            prompt=prompt,
                                            temperature=0,
                                            max_tokens=self.max_tokens
                                            )

        if self.logit_bias:
            def query_func():
                return openai.Completion.create(engine=self.model_name,
                                                prompt=prompt,
                                                temperature=0,
                                                max_tokens=self.max_tokens,
                                                logit_bias=self.logit_bias
                                                )

        if 'gpt-4' in self.model_name:
            query_messages = {
                "role": "user", "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }

            if len(image_paths) > 0:
                # import numpy as np
                # import cv2
                for img_p in image_paths:

                    # img = np.zeros((480, len(image_paths)*660, 3), dtype=np.uint8)
                    #
                    # for i in range(len(image_paths)):
                    #     img_t = cv2.imread(image_paths[i])
                    #     img[:, 660*i:660*i+640, :] = img_t
                    #
                    # img_p = "/home/vincent/py-pro/AirVLN-main/AirVLN/files/rgb_obs.png"
                    # cv2.imwrite(img_p, img)
                    base64_image = self.encode_image(img_p)
                    query_messages["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    })


            # print(query_messages)
            def query_func():
                completion = openai.ChatCompletion.create(model=self.model_name,
                                                          messages=[query_messages],
                                                          max_tokens=self.max_tokens,
                                                          temperature=0,
                                                          )
                text = completion['choices'][0]['message']['content']
                if text[0] != ' ':
                    text = ' ' + text
                completion['choices'][0]['text'] = text
                return completion

        try:
            response = query_func()
        except Exception as e:
            print(e)
            self.save_cache()
            time.sleep(10)
            print('try again')
            return self.query_api(prompt)

        if show_response:
            print('API Response:')
            print(response)
            print('')
        self.queried_tokens += response['usage']['total_tokens']
        # sequence = prompt + response['choices'][0]['text']
        sequence = response['choices'][0]['text']
        assert response['choices'][0]['finish_reason'] in self.finish_reasons + [None]
        return sequence

    def query_api(self, prompt, image_path=None,system=None, show_response=True):

        def query_func():
            return openai.Completion.create(engine=self.model_name,
                                            prompt=prompt,
                                            temperature=0,
                                            max_tokens=self.max_tokens
                                            )

        if self.logit_bias:
            def query_func():
                return openai.Completion.create(engine=self.model_name,
                                                prompt=prompt,
                                                temperature=0,
                                                max_tokens=self.max_tokens,
                                                logit_bias=self.logit_bias
                                                )

        if 'gpt-4' in self.model_name:
            query_messages = {
                "role": "user", "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }

            if image_path is not None:
                base64_image = self.encode_image(image_path)
                query_messages["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })

            # print(query_messages)
            def query_func():
                completion = openai.ChatCompletion.create(model=self.model_name,
                                                          messages=[query_messages],
                                                          max_tokens=self.max_tokens,
                                                          temperature=0,
                                                          )
                text = completion['choices'][0]['message']['content']
                if text[0] != ' ':
                    text = ' ' + text
                completion['choices'][0]['text'] = text
                return completion

        try:
            response = query_func()
        # except (openai.APIError,
        #         openai.RateLimitError,
        #         openai.APIConnectionError) as e:
        except Exception as e:
            print(e)
            self.save_cache()
            time.sleep(10)
            print('try again')
            return self.query_api(prompt)

        if show_response:
            print('API Response:')
            print(response)
            print('')
        self.queried_tokens += response['usage']['total_tokens']
        # sequence = prompt + response['choices'][0]['text']
        sequence = response['choices'][0]['text']
        assert response['choices'][0]['finish_reason'] in self.finish_reasons + [None]
        return sequence

    def query_api_map_gpt(self, prompt, system=None, image_path=None, show_response=False):
        def query_func():
            return openai.Completion.create(engine=self.model_name,
                                            prompt=prompt,
                                            temperature=0,
                                            max_tokens=self.max_tokens
                                            )

        if self.logit_bias:
            def query_func():
                return openai.Completion.create(engine=self.model_name,
                                                prompt=prompt,
                                                temperature=0,
                                                max_tokens=self.max_tokens,
                                                logit_bias=self.logit_bias
                                                )

        if 'gpt-4' in self.model_name:
            query_messages = [
                {
                    "role": "system",
                    "content": system
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            # if image_path is not None:
            #     base64_image = self.encode_image(image_path)
            #     query_messages["content"].append({
            #         "type": "image_url",
            #         "image_url": f"data:image/jpeg;base64,{base64_image}"
            #     })

            # print(query_messages)
            def query_func():
                completion = openai.ChatCompletion.create(model=self.model_name,
                                                          messages=query_messages,
                                                          max_tokens=self.max_tokens,
                                                          temperature=0,
                                                          )
                text = completion['choices'][0]['message']['content']
                if text[0] != ' ':
                    text = ' ' + text
                completion['choices'][0]['text'] = text
                return completion

        try:
            response = query_func()
        except (openai.error.APIStatusError,
                openai.error.RateLimitError,
                openai.error.APIConnectionError) as e:
            print(e)
            self.save_cache()
            time.sleep(10)
            print('try again')
            return self.query_api(prompt)

        if show_response:
            print('API Response:')
            print(response)
            print('')
        self.queried_tokens += response['usage']['total_tokens']
        # sequence = prompt + response['choices'][0]['text']
        sequence = response['choices'][0]['text']
        assert response['choices'][0]['finish_reason'] in self.finish_reasons + [None]
        return sequence


# if you want to use OPenAI_LLM_v3, you need upgrade your openai to version: pip install openai==1.58.1
class OpenAI_LLM_v3(LLM):
    def __init__(self, model_name, api_key, client_type="openai", logit_bias=None, max_tokens=64, finish_reasons=None, **kwargs):

        if client_type == "openai":
            self.client = OpenAI(
                api_key=api_key,
            )
        elif client_type == "Azure":
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version="2024-07-01-preview",
                azure_endpoint="https://zhangweichen-3d-gpt4o.openai.azure.com/"
            )
        self.logit_bias = logit_bias

        self.finish_reasons = finish_reasons
        if finish_reasons is None:
            self.finish_reasons = ['stop', 'length']

        super().__init__(api_key, model_name, max_tokens, **kwargs)

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def query_apis(self, prompt, image_paths=[], show_response=True):
        def query_func():
            query_messages = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }

            if len(image_paths) > 0:
                for img_p in image_paths:
                    base64_image = self.encode_image(img_p)
                    query_messages["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    })

            completion = self.client.chat.completions.create(
                messages=[
                    query_messages
                ],
                model="gpt-4o",
            )

            message = completion.choices[0].message
            content = unicodedata.normalize('NFKC', message.content)

            return content

        try:
            response = query_func()
        except Exception as e:
            print(e)
            self.save_cache()
            time.sleep(10)
            print('try again')
            return self.query_api(prompt)

        if show_response:
            print('API Response:')
            print(response)
            print('')

        return response

    def query_api(self, prompt, image_path=None,system=None, show_response=True):

        def query_func():
            query_messages = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }

            if image_path is not None:
                base64_image = self.encode_image(image_path)
                query_messages["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })

            completion = self.client.chat.completions.create(
                messages=[
                    query_messages
                ],
                model="gpt-4o",
            )

            message = completion.choices[0].message
            content = unicodedata.normalize('NFKC', message.content)

            return content

        try:
            response = query_func()

        except Exception as e:
            print(e)
            self.save_cache()
            time.sleep(10)
            print('try again')
            return self.query_api(prompt)

        if show_response:
            print('API Response:')
            print(response)
            print('')

        return response

    def query_api_map_gpt(self, prompt, system=None, image_path=None, show_response=False):

        def query_func():
            query_messages = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }

            sys_query_messages = {
                "role": "system",
                "content": system
            }

            completion = self.client.chat.completions.create(
                messages=[
                    sys_query_messages, query_messages
                ],
                model="gpt-4o",
            )

            message = completion.choices[0].message
            content = unicodedata.normalize('NFKC', message.content)

            return content

        try:
            response = query_func()
        except Exception as e:
            print(e)
            self.save_cache()
            time.sleep(10)
            print('try again')
            return self.query_api(prompt)

        if show_response:
            print('API Response:')
            print(response)
            print('')
        return response


class Ollama_LLM(LLM):
    def __init__(self, model_name, api_key, logit_bias=None, max_tokens=64, finish_reasons=None, **kwargs):
        client = OpenAI(
            base_url='http://localhost:11434/v1/',
            api_key='ollama'  # API key is required but ignored here
        )

        self.logit_bias = logit_bias

        super().__init__(api_key, model_name, max_tokens, **kwargs)

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def query_apis(self, prompt, image_paths=[], show_response=True):
        def query_func():
            query_messages = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }

            if len(image_paths) > 0:
                for img_p in image_paths:
                    base64_image = self.encode_image(img_p)
                    query_messages["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    })

            completion = self.client.chat.completions.create(
                messages=[
                    query_messages
                ],
                model="gpt-4o",
            )

            message = completion.choices[0].message
            content = unicodedata.normalize('NFKC', message.content)

            return content

        try:
            response = query_func()
        except Exception as e:
            print(e)
            self.save_cache()
            time.sleep(10)
            print('try again')
            return self.query_api(prompt)

        if show_response:
            print('API Response:')
            print(response)
            print('')

        return response

    def query_api(self, prompt, image_path=None,system=None, show_response=True):

        def query_func():
            query_messages = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }

            if image_path is not None:
                base64_image = self.encode_image(image_path)
                query_messages["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })

            completion = self.client.chat.completions.create(
                messages=[
                    query_messages
                ],
                model=self.model_name,
            )

            message = completion.choices[0].message
            content = unicodedata.normalize('NFKC', message.content)

            return content

        try:
            response = query_func()

        except Exception as e:
            print(e)
            self.save_cache()
            time.sleep(10)
            print('try again')
            return self.query_api(prompt)

        if show_response:
            print('API Response:')
            print(response)
            print('')

        return response


class OpenAI_ChatLLM(LLM):

    def __init__(self, model_name, api_key, logit_bias=None, max_tokens=64, finish_reasons=None, **kwargs):
        openai.api_key = api_key
        self.logit_bias = logit_bias

        self.finish_reasons = finish_reasons
        if finish_reasons is None:
            self.finish_reasons = ['stop', 'length']

        super().__init__(api_key, model_name, max_tokens, **kwargs)

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def query_apis(self, prompt, image_paths=[], show_response=True):
        def query_func():
            return openai.ChatCompletion.create(engine=self.model_name,
                                            prompt=prompt,
                                            temperature=0,
                                            max_tokens=self.max_tokens
                                            )

        if self.logit_bias:
            def query_func():
                return openai.ChatCompletion.create(engine=self.model_name,
                                                prompt=prompt,
                                                temperature=0,
                                                max_tokens=self.max_tokens,
                                                logit_bias=self.logit_bias
                                                )

        if 'gpt-4' in self.model_name:
            query_messages = {
                "role": "user", "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }

            if len(image_paths) > 0:
                import numpy as np
                import cv2
                img = np.zeros((480, len(image_paths)*660, 3), dtype=np.uint8)

                for i in range(len(image_paths)):
                    img_t = cv2.imread(image_paths[i])
                    img[:, 660*i:660*i+640, :] = img_t

                img_p = "/home/vincent/py-pro/AirVLN-main/AirVLN/files/rgb_obs.png"
                cv2.imwrite(img_p, img)
                base64_image = self.encode_image(img_p)
                query_messages["content"].append({
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    })

                # for image_path in image_paths:
                #     base64_image = self.encode_image(image_path)
                #     query_messages["content"].append({
                #         "type": "image_url",
                #         "image_url": f"data:image/jpeg;base64,{base64_image}"
                #     })

            print(query_messages)
            def query_func():
                completion = openai.ChatCompletion.create(model=self.model_name,
                                                          messages=[query_messages],
                                                          max_tokens=self.max_tokens,
                                                          temperature=0,
                                                          )
                text = completion['choices'][0]['message']['content']
                if text[0] != ' ':
                    text = ' ' + text
                completion['choices'][0]['text'] = text
                return completion

        try:
            response = query_func()
        except (openai.error.ServiceUnavailableError,
                openai.error.RateLimitError,
                openai.error.ServiceUnavailableError,
                openai.error.APIConnectionError) as e:
            print(e)
            self.save_cache()
            time.sleep(10)
            print('try again')
            return self.query_api(prompt)

        if show_response:
            print('API Response:')
            print(response)
            print('')
        self.queried_tokens += response['usage']['total_tokens']
        # sequence = prompt + response['choices'][0]['text']
        sequence = response['choices'][0]['text']
        assert response['choices'][0]['finish_reason'] in self.finish_reasons + [None]
        return sequence


    def query_api(self, prompt, image_path=None, show_response=True):

        def query_func():
            return openai.ChatCompletion.create(model=self.model_name,
                                                messages=[
                                                    {"role": "user", "content": prompt}
                                                ],
                                                max_tokens=self.max_tokens
                                            )

        if self.logit_bias:
            def query_func():
                return openai.ChatCompletion.create(engine=self.model_name,
                                                prompt=prompt,
                                                temperature=0,
                                                max_tokens=self.max_tokens,
                                                logit_bias=self.logit_bias
                                                )

        if 'gpt-4' in self.model_name:
            query_messages = {
                "role": "user", "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }

            if image_path is not None:
                base64_image = self.encode_image(image_path)
                query_messages["content"].append({
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
                })

            # print(query_messages)
            def query_func():
                completion = openai.ChatCompletion.create(model=self.model_name,
                                                          messages=[query_messages],
                                                          max_tokens=self.max_tokens,
                                                          temperature=0,
                                                          )
                text = completion['choices'][0]['message']['content']
                if text[0] != ' ':
                    text = ' ' + text
                completion['choices'][0]['text'] = text
                return completion

        try:
            response = query_func()
        except (openai.error.ServiceUnavailableError,
                openai.error.RateLimitError,
                openai.error.ServiceUnavailableError,
                openai.error.APIConnectionError) as e:
            print(e)
            self.save_cache()
            time.sleep(10)
            print('try again')
            return self.query_api(prompt)

        if show_response:
            print('API Response:')
            print(response)
            print('')
        self.queried_tokens += response['usage']['total_tokens']
        # sequence = prompt + response['choices'][0]['text']
        sequence = response['choices'][0].message['content']
        assert response['choices'][0]['finish_reason'] in self.finish_reasons + [None]
        return sequence


if __name__ == "__main__":
    pass