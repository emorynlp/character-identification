import json

from structure.nodes import TokenNode
from experiments.baseline.tools.mention import SingEvalMentionNode
from structure.transcripts import Utterance, Scene, Episode
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
        utterances = []
        scene_mentions = []

        for i, utterance_json in enumerate(utterance_jsons):
            prev_speakers = reversed([u.speakers for u in utterances[:i]])
            utterance, mentions = self.read_utterance_json(utterance_json, prev_speakers)
            utterances.append(utterance)
            scene_mentions.extend(mentions)

        # remove any entities which do not have sing. mentions references but have pl. mention references
        sing_labels = set([m.gold_ref for m in scene_mentions])
        for m in scene_mentions:
            if m.plural:
                # if entity does not exist in the singular labels, then replace it with "#other#" label
                m.all_gold_refs = [gref
                                   if gref in sing_labels or gref == "#other#" or gref == "#general#"
                                   else "#other#"
                                   for gref in m.all_gold_refs]

                # remove any duplicate labels
                m.all_gold_refs = list(set(m.all_gold_refs))

            if len(m.all_gold_refs) == 0:
                m.all_gold_refs = ["#other#"]

        season_mentions.extend(scene_mentions)

        for i in range(len(utterances) - 1):
            utterances[i + 1]._previous = utterances[i]
            utterances[i]._next = utterances[i + 1]

        return Scene(scene_num, utterances)

    def read_utterance_json(self, utterance_json, prev_speakers):
        speakers = utterance_json["speakers"]

        word_forms = utterance_json["tokens"]
        pos_tags = utterance_json["part_of_speech_tags"]
        dep_tags = utterance_json["dependency_tags"]
        dep_heads = utterance_json["dependency_heads"]
        ner_tags = utterance_json["named_entity_tags"]
        ref_tags = utterance_json["character_entities"]

        tokens_all = self.parse_token_nodes(word_forms, pos_tags, dep_tags, dep_heads, ner_tags)
        utterance_mentions = self.parse_mention_nodes(tokens_all, ref_tags, prev_speakers)

        return Utterance(speakers, statements=tokens_all), utterance_mentions

    def parse_token_nodes(self, word_forms, pos_tags, dep_tags, dep_heads, ner_tags):
        tokens_all = []

        # sentence
        for word_s, pos_s, dep_s, h_dep_s, ner_s in zip(word_forms, pos_tags, dep_tags, dep_heads, ner_tags):
            tokens = []

            for idx, word, pos, dep, ner in zip(range(len(word_s)), word_s, pos_s, dep_s, ner_s):
                token = TokenNode(idx, word, pos, ner, dep)
                tokens.append(token)

            if len(h_dep_s) != len(tokens):
                print(word_forms)
                print(h_dep_s)

            for idx, hid in enumerate(h_dep_s):
                tokens[idx].dep_head = tokens[hid - 1] if hid > 0 else None

            tokens_all.append(tokens)

        return tokens_all

    def parse_mention_nodes(self, tokens, referents, prev_speakers):
        mentions = []
        for token_s, ref_s in zip(tokens, referents):
            # condensed referent
            for condensed_mrefs in ref_s:
                ref = ""
                si, ei = condensed_mrefs[0], condensed_mrefs[1]
                mrefs = condensed_mrefs[2:]

                # remove mentions labeled Non-Entity b/c they do not refer to characters
                if mrefs == ["Non-Entity"]:
                    continue

                is_plural = True if len(mrefs) > 1 else False

                # remove General label from the referents of plural mentions
                if len(mrefs) > 1:
                    mrefs = list(set([mref if mref != "#GENERAL#" else "#OTHER#" for mref in mrefs]))

                # converts plural mentions to singular-esque mentions (i.e. all mentions have one referent)
                # rather than blindly pick one, pick the referent who is also the closest previous speaker
                if len(mrefs) == 1:
                    ref = mrefs[0]
                elif len(mrefs) > 1:
                    for spks in prev_speakers:
                        rset = set(mrefs).intersection(set(spks))
                        if len(rset) > 0:
                            ref = rset.pop()
                            break

                    if ref == "":
                        ref = mrefs[0]

                mention = SingEvalMentionNode(self.mid, token_s[si:ei], mrefs, ref, plural=is_plural)
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
