
# dictionary with stats for topics
topic0 = {'sex' : {},
          'age' : {},
          'ethnic' : {},
          'race' : {},
          'virus' : dict()}
topic1 = {'sex' : {},
          'age' : {},
          'ethnic' : dict(),
          'race' : dict(),
          'virus' : dict()}

with open('../mapping.txt', 'r') as f:
    for line in f:
        filename, topic = line.split(':::')
        # topic looks like this: Topic: [(0, 0.31863713), (1, 0.68136287)]
        # get 0 or 1, the biggest number
        topic = topic.split('[')[1]
        topicparts = topic.split('(')
        if len(topicparts) > 2:
            groep1str = float(topicparts[2].split(',')[1].split(')')[0].strip())
            groep0str = float(topicparts[1].split(',')[1].split(')')[0].strip())
            if groep1str > groep0str:
                topic_id = 1
            else:
                topic_id = 0
        else:
            topic_id = int(topicparts[1].split(',')[0].strip())
        filename = filename.split('_')[1]
        with open('processed_files/metadata_merged.tsv', 'r') as f2:
            for line2 in f2:
                split_line = line2.split('\t')
                if split_line[0] == filename:
                    # topic_id 0, write top topic0
                    sex = split_line[12]
                    age = split_line[11]
                    ethnic = split_line[13]
                    race = split_line[14]
                    virus = split_line[15]
                    if topic_id == 0:
                        topic0['sex'][sex] = topic0['sex'].get(sex, 0) + 1

                        topic0['age'][age] = topic0['age'].get(age, 0) + 1
                        topic0['ethnic'][ethnic] = topic0['ethnic'].get(ethnic, 0) + 1
                        topic0['race'][race] = topic0['race'].get(race, 0) + 1
                        topic0['virus'][virus] = topic0['virus'].get(virus, 0) + 1
                    else:
                        
                        topic1['sex'][sex] = topic1['sex'].get(sex, 0) + 1
                        topic1['age'][age] = topic1['age'].get(age, 0) + 1
                        topic1['ethnic'][ethnic] = topic1['ethnic'].get(ethnic, 0) + 1
                        topic1['race'][race] = topic1['race'].get(race, 0) + 1
                        topic1['virus'][virus] = topic1['virus'].get(virus, 0) + 1


                    # topic_id 1, write top topic1
                    break
import pprint
print("Topic 0")
pprint.pprint(topic0)
print("Topic 1")
pprint.pprint(topic1)
