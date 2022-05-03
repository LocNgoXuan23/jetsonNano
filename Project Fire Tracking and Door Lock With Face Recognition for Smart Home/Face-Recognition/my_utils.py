import yaml

def getConfig(yaml_file='./data/config.yml'):
	with open(yaml_file, 'r') as file:
		cfgs = yaml.load(file, Loader=yaml.FullLoader)
	return cfgs

def updateConfig(yaml_file='./data/config.yml'):
	cfgs = getConfig(yaml_file)
	cfgs['currentMember'] += 1

	with open(yaml_file, 'w') as file:
		yaml.dump(cfgs, file)

