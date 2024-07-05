#!/bin/bash

set -e

max_jobs=6

# All the ones I looked at initially
# All wildtype
fish_ids=(40 42 47 53 61 70 89 95 99 201 205 218 224 230 240 243 247 363 372 373 384 389 441 490 545 575)

# The ones I'm using to fine-tune the model
test_fish=(40 95 218 247 372 389 53 230 545)
for i in "${fish_ids[@]}";
do
	# If any files called cropped/i/sub_window* exist, skip
	if compgen -G "cropped/$i/sub_window*" > /dev/null; then
		echo "Skipping $i"
		continue
	fi

	(time python localisation/crop_head.py $i --plot) &

	if (( $(jobs -r | wc -l) >= $max_jobs )) ; then
		# wait until there is one or less
		wait -n
	fi
done

# Wait for all jobs to finish
wait