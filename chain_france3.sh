cd "/c/Users/Lenovo/Downloads/Projects/ML Hackathon"
until grep -q "FRANCE DONE\|FAILED" run_france2.out 2>/dev/null; do sleep 30; done
bash run_france3.sh >> run_france3.out 2>&1
