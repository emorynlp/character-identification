from experiments.latest.tools.mention import other, general


def test_plural_batch_fidelity(reconstructed, gold):
    dkeys = reconstructed.keys()
    gkeys = gold.keys()

    dkeyset = set(dkeys)
    gkeyset = set(gkeys)

    if dkeyset != gkeyset and len(dkeys) != len(gkeys):
        raise Exception("Gold mention to clusters map and dynamic mention to clusters map do not have identical mention keys.\n" +
                        "Mention keys expected but not found: %s\n" % str(gkeyset.difference(dkeyset)) +
                        "Mention keys found but not expected: %s\n" % str(dkeyset.difference(gkeyset)))
    else:
        for key in dkeys:
            dcs = reconstructed[key]
            gcs = gold[key]

            if len(dcs) != len(gcs):
                print(dcs)
                print(gcs)
                print([[m.gold_refs for m in dc] for dc in dcs])
                print([[m.gold_refs for m in gc] for gc in gcs])
                raise Exception("Key %s does not have correct number of clusters\n" % key)

            if len(dcs) == 1:
                dc = set(dcs[0])
                gc = set(gcs[0])

                if dc != gc:
                    print(dc)
                    print(gc)
                    raise Exception("Key %s does not have correct cluster\n" % key +
                                    "Mentions expected but not found: %s\n" % str(gc.difference(dc)) +
                                    "Mentions found but not expected: %s\n" % str(dc.difference(gc)))


def reconstruct_state_clusters(preds, state):
    m2dcs = {}
    mset = set()

    mentions = [other, general] + state

    mpairs = []
    for idx, m in enumerate(mentions[2:], 2):
        for a in mentions[:idx]:
            mpairs.append((m, a))

    for pred, (a, m) in zip(preds, mpairs):
        # predict antecedent needs to create new cluster
        if pred == 1:
            if a.is_other() or a.is_general():
                mc = {m}
                if m in m2dcs:
                    m2dcs[m].append(mc)
                else:
                    m2dcs[m] = [mc]
            else:
                if a in m2dcs:
                    ante_cs = m2dcs[a]
                    ante_cs[0].add(m)

                    if m in m2dcs and ante_cs[0] not in m2dcs[m]:
                        m2dcs[m].append(ante_cs[0])
                    elif m not in m2dcs:
                        m2dcs[m] = [ante_cs[0]]
                else:
                    mc = {a, m}
                    m2dcs[a] = [mc]

                    if m in m2dcs:
                        m2dcs[m].append(mc)
                    else:
                        m2dcs[m] = [mc]

            mset.add(m)
            mset.add(a)
        # predict current mention needs to create new cluster
        elif pred == 2:
            if m in m2dcs:
                m2dcs[m][0].add(a)

                if a in m2dcs and m2dcs[m][0] not in m2dcs[a]:
                    m2dcs[a].append(m2dcs[m][0])
                elif a not in m2dcs:
                    m2dcs[a] = [m2dcs[m][0]]
            else:
                mc = {m, a}
                m2dcs[m] = [mc]

                if a in m2dcs:
                    m2dcs[a].append(mc)
                else:
                    m2dcs[a] = [mc]

            mset.add(m)
            mset.add(a)

    diffset = set(state) - mset

    for dm in diffset:
        m2dcs[dm] = [{dm}]

    test_plural_batch_fidelity(m2dcs, state.m2_gCs)
