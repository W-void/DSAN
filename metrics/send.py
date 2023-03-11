# !/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import socket
import base64
import hashlib
import requests
import hmac
import time

PUB_ID = '137550031793'
SINGLE_URL = 'https://xmapi.vip.sankuai.com/api/pub/push'
GROUP_URL = 'https://xmapi.vip.sankuai.com/api/pub/pushToRoom'
pub_id2info = {
	'137442753240': ['881100608x42o011', '862fd76504c94c2991a0e7f0b5842459'],
	'137550031793': ['1234002Y3391n710', '3ea9dc7933a215fd7afcbac9bc2c88be'],
	'137820335797': ['122000439812C22f', '521589689f9a87b2c7d34a6dbb7c5579']
}


# 大象官方消息签名验证
def gen_headers(client_id, client_secret, url_path, http_method):
	timestamp = time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.gmtime())
	string_to_sign = ('%s %s\n%s' % (http_method, url_path, timestamp))
	hmac_bytes = hmac.new(bytes(client_secret.encode('ascii')),
	                      bytes(string_to_sign.encode('ascii')),
	                      hashlib.sha1).digest()
	auth = base64.b64encode(hmac_bytes).decode("utf-8")

	return {
		'Date': timestamp,
		'Authorization': 'MWS %s:%s' % (client_id, auth),
		'Content-Type': 'application/json;charset=utf-8',
	}


def gen_dx_message(from_uid, dx_message, receivers):
	single_data = {
		'fromUid': from_uid,
		'receivers': receivers,
		'messageType': 'text',
		'body': {
			'text': dx_message
		}
	}

	return single_data


def gen_dx_group_message(from_uid, dx_message, group_id):
	group_data = {
		'fromUid': from_uid,
		'toGid': group_id,
		'messageType': 'text',
		'body': {
			'text': dx_message
		}
	}

	return group_data


# 大象发送接口
def send_dx_message(dx_message, receivers, pub_id=PUB_ID):
	"""
	发送个人消息
	:param dx_message: msg
	:param receivers: 接收人mis号list
	:param pub_id: 大象公众号id（默认点评增长算法组机器人）
	:return:
	"""
	client_id, client_secret = pub_id2info[pub_id]
	# print(dx_message)
	dx_headers = gen_headers(client_id, client_secret, '/api/pub/push', 'PUT')
	dx_data = gen_dx_message(pub_id, dx_message, receivers)
	dx_response = requests.put(SINGLE_URL, headers=dx_headers, data=json.dumps(dx_data), verify=False)


# print(u'大象个人API返回的发送结果：', dx_response.content.decode('utf-8'))


# 群组接口，首先要向大象平台申请向相应群组ID发送大象Message的权限，默认不能发送
# 申请见https://km.sankuai.com/page/42249861
def send_dx_group_message(dx_message, group_id, pub_id=PUB_ID):
	"""
	发送群组消息
	:param dx_message: msg
	:param group_id: 群id
	:param pub_id: 大象公众号id（默认点评增长算法组机器人）
	:return:
	"""
	client_id, client_secret = pub_id2info[pub_id]
	dx_group_headers = gen_headers(client_id, client_secret, '/api/pub/pushToRoom', 'PUT')
	dx_group_data = gen_dx_group_message(pub_id, dx_message, group_id)
	dx_group_response = requests.put(GROUP_URL, headers=dx_group_headers, data=json.dumps(dx_group_data), verify=False)
	print(u"大象个人API返回的发送结果：", dx_group_response.content)


if __name__ == '__main__':
	from send import *

	send_dx_group_message('hello world', '66638842359', '137820335797')
	send_dx_message('hello world', ['dongjian03'], '137820335797')
