import json
from pydash import flatten

from experiments.latest.tools.mention import PluralMentionNode
from structure import Utterance, Scene, Episode, TokenNode
from util import idutils


class SpliceReader:
    def __init__(self):
        self.mid = 0

    def read_season_json(self, json_path):
        season_mentions = []

        with open(json_path, "r") as fin:
            season_json = json.load(fin)
            episode_jsons = season_json["episodes"]
            episodes = [self.read_episode_json(episode_json, season_mentions)
                        for episode_json in episode_jsons]

            for i in range(len(episodes) - 1):
                episodes[i + 1]._previous = episodes[i]
                episodes[i]._next = episodes[i + 1]

            self.assign_metadata(episodes)

        return episodes, season_mentions

    def read_episode_json(self, episode_json, season_mentions):
        episode_id = episode_json["episode_id"]
        episode_num = idutils.parse_episode_id(episode_id)[-1]

        scene_jsons = episode_json["scenes"]
        scenes = [self.read_scene_json(scene_json, season_mentions)
                  for scene_json in scene_jsons]

        for i in range(len(scenes) - 1):
            scenes[i + 1]._previous = scenes[i]
            scenes[i]._next = scenes[i + 1]

        return Episode(episode_num, scenes)

    def read_scene_json(self, scene_json, season_mentions):
        scene_id = scene_json["scene_id"]
        scene_num = idutils.parse_scene_id(scene_id)[-1]

        utterance_jsons = scene_json["utterances"]
        utterance_mention_pairs = [self.read_utterance_json(utterance_json) for utterance_json in utterance_jsons]

        utterances = [pair[0] for pair in utterance_mention_pairs]
        scene_mentions = flatten([pair[1] for pair in utterance_mention_pairs])

        # remove any entities which do not have sing. mentions references but have pl. mention references
        sing_labels = set([m.gold_refs[0] for m in scene_mentions if not m.plural])
        for m in scene_mentions:
            if m.plural:
                # if entity does not exist in the singular labels, then replace it with "#other#" label
                m.gold_refs = [gref
                               if gref in sing_labels or gref == "#other#" or gref == "#general#"
                               else "#other#"
                               for gref in m.gold_refs]

                # remove any duplicate labels
                m.gold_refs = list(set(m.gold_refs))

            if len(m.gold_refs) == 0:
                m.gold_refs = ["#other#"]

        season_mentions.extend(scene_mentions)

        for i in range(len(utterances) - 1):
            utterances[i + 1]._previous = utterances[i]
            utterances[i]._next = utterances[i + 1]

        return Scene(scene_num, utterances)

    def read_utterance_json(self, utterance_json):
        speakers = utterance_json["speakers"]

        word_forms = utterance_json["tokens"]
        pos_tags = utterance_json["part_of_speech_tags"]
        dep_tags = utterance_json["dependency_tags"]
        dep_heads = utterance_json["dependency_heads"]
        ner_tags = utterance_json["named_entity_tags"]
        ref_tags = utterance_json["character_entities"]

        tokens_all = self.parse_token_nodes(word_forms, pos_tags, dep_tags, dep_heads, ner_tags)
        utterance_mentions = self.parse_mention_nodes(tokens_all, ref_tags)

        return Utterance(speakers, statements=tokens_all), utterance_mentions

    def parse_token_nodes(self, word_forms, pos_tags, dep_tags, dep_heads, ner_tags):
        tokens_all = []

        # sentence
        for word_s, pos_s, dep_s, h_dep_s, ner_s in zip(word_forms, pos_tags, dep_tags, dep_heads, ner_tags):
            tokens = []

            for idx, word, pos, dep, ner in zip(range(len(word_s)), word_s, pos_s, dep_s, ner_s):
                token = TokenNode(idx, word, pos, ner, dep)
                tokens.append(token)

            for idx, hid in enumerate(h_dep_s):
                tokens[idx].dep_head = tokens[hid - 1] if hid > 0 else None

            tokens_all.append(tokens)

        return tokens_all

    def parse_mention_nodes(self, tokens, referents):
        mentions = []
        for token_s, ref_s in zip(tokens, referents):
            # condensed referent
            for ref_cond in ref_s:
                # ref_cond = [<start index>, <end index>, <label 1>, <label 2>, ...]
                start_idx, end_idx = ref_cond[0], ref_cond[1]
                refs = list(set(ref_cond[2:]))

                if refs == ["Non-Entity"]:
                    continue

                is_plural = True if len(refs) > 1 else False

                # remove general label from plural mentions
                if len(refs) > 1:
                    refs = list(set([ref if ref != "#GENERAL#" else "#OTHER#" for ref in refs]))

                mention = PluralMentionNode(self.mid, token_s[start_idx:end_idx], refs, plural=is_plural)
                mentions.append(mention)

        return mentions

    def assign_metadata(self, episodes):
        for episode in episodes:
            for scene in episode.scenes:
                scene._episode = episode

                for utterance in scene.utterances:
                    utterance._scene = scene

                    for sentence in utterance.statements:
                        for token in sentence:
                            token._episode = episode
                            token._scene = scene
                            token._utterance = utterance


class StateWriter(object):
    def __init__(self):
        self.fout = None

    def open_file(self, file_path):
        self.fout = open(file_path, "w")

    def write_states(self, states):
        self.fout.write("Mention/Gold/System\n\n")

        for s in states:
            self.write_state(s)
        self.fout.close()

    def write_state(self, state):
        for m in state:
            result = "%s - %s / %s\n" % (str(m), str(m.gold_refs), str(m.auto_refs))
            self.fout.write(result)
