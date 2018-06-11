import re

sid_regex = re.compile("s(\d+).*")
eid_regex = re.compile("s(\d+)_e(\d+).*")
cid_regex = re.compile("s(\d+)_e(\d+)_c(\d+).*")
uid_regex = re.compile("s(\d+)_e(\d+)_c(\d+)_u(\d+).*")


def parse_season_id(id_string):
    return parse_id(id_string, sid_regex, 1)


def parse_episode_id(id_string):
    return parse_id(id_string, eid_regex, 2)


def parse_scene_id(id_string):
    return parse_id(id_string, cid_regex, 3)


def parse_utterance_id(id_string):
    return parse_id(id_string, uid_regex, 4)


def parse_id(id_string, id_regex, group_num):
    id_matcher = id_regex.match(id_string)

    if id_matcher:
        return [int(id_matcher.group(i)) for i in range(1, group_num + 1)]
    else:
        return []
