# Character Identification

Character Identification is an entity linking task that finds the global entity of each personal mention in multiparty dialogue. 
Let a mention be a nominal referring to a person (e.g., *she*, *mom*, *Judy*), and an entity be a character in a dialogue. 
The goal is to assign each mention to its entity, who may or may not participate in the dialogue. 
For the following example, the mention "mom" is not one of the speakers; nonetheless, it clearly refers to the specific person, *Judy Geller*, that could appear in some other dialogue. Identifying such mentions as real characters requires cross-document entity resolution, which makes this task challenging.

![Character Identification Example](http://nlp.mathcs.emory.edu/character-mining/img/character-identification-example.png)

This task is a part of the [Character Mining](https://github.com/emorynlp/character-mining) project led by the [Emory NLP](http://nlp.mathcs.emory.edu) research group.

## Dataset

Given the transcripts provided by the [Character Mining](https://github.com/emorynlp/character-mining) project, all personal mentions are annotated with their global entities.
For the above example, the first mention "I" is annotated with its global entity, *Ross Geller*, and the second mention "mom" is annotated with, *Judy Geller*, and so on.
The mention detection is first performed automatically then corrected manually.
The entity annotation is mostly crowdsourced although lots of them are fixed manually by experts.

* Latest release: v2.0 (TBA).
* [Release notes](doc/release-notes.md).

## Statistics

| Dataset | Episodes | Scenes | Utterances |  Tokens | Speakers | Mentions | Entities |
|:-------:|---------:|-------:|-----------:|--------:|---------:|---------:|---------:|
| TRN   | 76 | 987   | 18,789 | 262,650 | 265 | 36,385 | 628 |
| DEV   | 8  | 122   | 2142   | 28523   | 48  | 3932   | 102 |
| TST   | 13 | 192   | 3,597  | 50,232  | 91  | 7,050  | 165 |
| Total | 97 | 1,301 | 24,528 | 341,405 | 331 | 47,367 | 781 |

## Annotation

Each utterance is split into sentences and all mentions in each sentence are annotated with their entities.

```json
{
  "utterance_id": "s01_e01_c01_u002",
  "speakers": ["Joey Tribbiani"],
  "transcript": "C'mon, you're going out with the guy! There's gotta be something wrong with him!",
  "tokens": [
    ["C'mon", ",", "you", "'re", "going", "out", "with", "the", "guy", "!"],
    ["There", "'s", "got", "ta", "be", "something", "wrong", "with", "him", "!"]
  ],
  "character_entities": [
    [[2, 3, "Monica Geller"], [8, 9, "Paul the Wine Guy"]],
    [[8, 9, "Paul the Wine Guy"]]
  ]
}
```

For the above example, the utterance is split into two sentences.
The first sentence, "*C'mon, you're going out with the guy!*", has two mentions, *you* and *guy*, that are linked to the entities *Monica Geller* and *Paul the Wine Guy*.
The second sentence, "*There's gotta be something wrong with him!*", includes one mention, *him*, that is linked to the entity, *Paul the Wine Guy*.
Each mention is annotated by the following format:

```
entity ::= [begin_index, end_index, entity_label]

begin_index ::= the beginning token index of the mention (inclusive)
end_index ::= the ending token index of the mention (exclusive)
entity_label ::= the label of the entity
```

## References

* [Robust Coreference Resolution and Entity Linking on Dialogues: Character Identification on TV Show Transcripts](http://www.aclweb.org/anthology/K17-1023), Henry Y. Chen, Ethan Zhou, and Jinho D. Choi. Proceedings of the 21st Conference on Computational Natural Language Learning, CoNLL'17, 216-225 Vancouver, Canada, 2017 ([slides](https://www.slideshare.net/jchoi7s/robust-coreference-resolution-and-entity-linking-on-dialogues-character-identification-on-tv-show-transcripts)).
* [Character Identification on Multiparty Conversation: Identifying Mentions of Characters in TV Shows](http://www.aclweb.org/anthology/W16-3612), Henry Y. Chen and Jinho D. Choi. Proceedings of the 17th Annual SIGdial Meeting on Discourse and Dialogue, SIGDIAL'16, 90-100 Los Angeles, CA, 2016 ([poster](https://www.slideshare.net/jchoi7s/character-identification-on-multiparty-conversation-identifying-mentions-of-characters-in-tv-shows)).

## Shared Task

* [SemEval 2018 Task 4: Character Identification on Multiparty Dialogues](https://github.com/emorynlp/semeval-2018-task4).

## Contact

* [Jinho D. Choi](http://www.mathcs.emory.edu/~choi).
