#!/bin/bash

set -e

# Some wildtypes
max_jobs=6
for i in 40 42 47 53 61 70 89 95 99 201 205 218 224 230 240 243 247 363 372 373 384 389 441 490 545 575;
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