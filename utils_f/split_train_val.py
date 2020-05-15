val = ['blackswan','bmx-trees','breakdance','camel','car-roundabout','car-shadow','cows','dance-twirl','dog','drift-chicane','drift-straight','goat','horsejump-high','kite-surf','libby','motocross-jump','paragliding-launch','parkour','scooter-black','soapbox']
train = ['bear', 'bmx-bumps','boat','breakdance-flare','bus','car-turn','dance-jump','dog-agility','drift-turn','elephant','flamingo','hike','hockey','horsejump-low','kite-walk','lucia','mallard-fly','mallard-water','motocross-bumps','motorbike','paragliding','rhino','rollerblade','scooter-gray','soccerball','stroller','surf','swing','tennis','train']
fets = ['tennis','swing','surf','stroller','soccerball','scooter-gray','rhino','paragliding','motocross-bumps','mallard-fly','flamingo','elephant','drift-turn','dog-agility','dance-jump','soapbox','parkour','scooter-black','paragliding-launch','motocross-jump','libby','kite-surf','horsejump-high','drift-straight','drift-chicane','car-roundabout','bear','boat','camel','hike','lucia','train','blackswan','breakdance','car-turn','hockey','mallard-water','bmx-bumps','breakdance-flare','cows','horsejump-low','motorbike','bmx-trees','bus','goat','kite-walk','rollerblade','car-shadow', 'dance-twirl', 'dog']

val_train = [*val, *train]
print("Training videos: ", len(train))
print("Validation videos: ", len(val))


for c in val_train:
    if c not in fets:
        if c in val:
            print(c, " validation")
        else:
            print(c, " training")