# Character Identification

Character Identification is an entity linking task that finds the global entity of each personal mention in multiparty dialogue. 
Let a mention be a nominal referring to a person (e.g., *she*, *mom*, *Judy*), and an entity be a character in a dialogue. 
The goal is to assign each mention to its entity, who may or may not participate in the dialogue. 
For the following example, the mention "mom" is not one of the speakers; nonetheless, it clearly refers to the specific person, *Judy Geller*, that could appear in some other dialogue. Identifying such mentions as real characters requires cross-document entity resolution, which makes this task challenging.

![Character Identification Example](http://nlp.mathcs.emory.edu/character-mining/img/character-identification-example.png)

This task is a part of the [Character Mining](../../../character-mining) project led by the [Emory NLP](http://nlp.mathcs.emory.edu) research group.

## Dataset

All personal mentions are annotated with their global entities.
For the above example, the first mention "I" is annotated with its global entity, *Ross Geller*, and the second mention "mom" is annotated with, *Judy Geller*, and so on.
The mention detection is first performed automatically then corrected manually.
The entity annotation is mostly crowdsourced although lots of them are fixed manually by experts.

* Latest release: [v2.0](https://github.com/emorynlp/character-identification/archive/character-identification-2.0.tar.gz).
* [Release notes](https://github.com/emorynlp/character-identification/releases).

## Statistics

For each season, episodes 1 ~ 19 are used for training (TRN), 20 ~ 21 for development (DEV), and 22 ~ rest for evaluation (TST).

| Dataset | Episodes | Scenes | Utterances |  Tokens | Speakers | Mentions | Entities |
|:-------:|---------:|-------:|-----------:|--------:|---------:|---------:|---------:|
| TRN   | 76 | 987   | 18,789 | 262,650 | 265 | 36,385 | 628 |
| DEV   | 8  | 122   | 2142   | 28523   | 48  | 3932   | 102 |
| TST   | 13 | 192   | 3,597  | 50,232  | 91  | 7,050  | 165 |
| Total | 97 | 1,301 | 24,528 | 341,405 | 331 | 47,367 | 781 |

## Annotation

Each utterance is split into sentences and personal mentions in every sentence are annotated with their entities.
For the example below, the utterance consists of one sentence including four mentions.
The first three mentions, *I*, **mom* and *dad*, are singular that refer to *Ross Geller*, *Judy Geller* and *Jack Geller*, respectively.
The last mention, *they*, is plural that refers to both *Judy Geller* and *Jack Geller*.

```json
{
  "utterance_id": "s01_e01_c01_u039",
  "speakers": ["Ross Geller"],
  "transcript": "I told mom and dad last night, they seemed to take it pretty well.",
  "tokens": [
    ["I", "told", "mom", "and", "dad", "last", "night", ",", "they", "seemed", "to", "take", "it", "pretty", "well", "."]
  ],
  "character_entities": [
    [[0, 1, "Ross Geller"], [2, 3, "Judy Geller"], [4, 5, "Jack Geller"], [8, 9, "Jack Geller", "Judy Geller"]]
  ]
}
```

Each mention is annotated by the following scheme:

```
[begin_index, end_index, entity(, entity)*]
```

* `begin_index: int` - the beginning token index of the mention (inclusive).
* `end_index: int` - the ending token index of the mention (exclusive).
* `entity: str` - the label of the entity.


## Citatioin

* [They Exist! Introducing Plural Mentions to Coreference Resolution and Entity Linking](http://aclweb.org/anthology/C18-1003). Ethan Zhou and Jinho D. Choi. In Proceedings of the 27th International Conference on Computational Linguistics, COLING'18, 2018 ([slides](https://www.slideshare.net/jchoi7s/they-exist-introducing-plural-mentions-to-coreference-resolution-and-entity-linking)). 

## References

* [Robust Coreference Resolution and Entity Linking on Dialogues: Character Identification on TV Show Transcripts](http://www.aclweb.org/anthology/K17-1023), Henry Y. Chen, Ethan Zhou, and Jinho D. Choi. Proceedings of the 21st Conference on Computational Natural Language Learning, CoNLL'17, 2017 ([slides](https://www.slideshare.net/jchoi7s/robust-coreference-resolution-and-entity-linking-on-dialogues-character-identification-on-tv-show-transcripts)).
* [Character Identification on Multiparty Conversation: Identifying Mentions of Characters in TV Shows](http://www.aclweb.org/anthology/W16-3612), Henry Y. Chen and Jinho D. Choi. Proceedings of the 17th Annual SIGdial Meeting on Discourse and Dialogue, SIGDIAL'16, 2016 ([poster](https://www.slideshare.net/jchoi7s/character-identification-on-multiparty-conversation-identifying-mentions-of-characters-in-tv-shows)).

## Shared Task

* [SemEval 2018 Task 4: Character Identification on Multiparty Dialogues](../../../semeval-2018-task4).

## Contact

* [Jinho D. Choi](http://www.mathcs.emory.edu/~choi).
