mkdir -p ./record
for trial in 1 2 3 4 5 6 7 8 9 10
do
	for source in DE007 DE014 DE021 FE007 FE014 FE021 DE FE
	do
		for target in DE007 DE014 DE021 FE007 FE014 FE021 DE FE
		do
			rm -r ./utils/__pycache__
			rm -r ./model/__pycache__
			task="$source-$target"
			# echo $task
			python main.py --source $source --target $target --task $task
			sleep 5
		done
	done
	path="./record/FRAN$trial"
	mkdir -p $path
	mv ./record/*.txt $path
done