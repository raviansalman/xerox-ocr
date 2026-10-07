#!/bin/bash
USER_ID="user_alpha"
FILE_URL="https://raw.githubusercontent.com/hwchase17/chat-your-data/master/state_of_the_union.txt"
FILE_ID="test_${USER_ID}_$(uuidgen | cut -c1-8)"

echo "Submitting $FILE_ID for $USER_ID..."
response=$(curl -s -X POST http://127.0.0.1:8000/process \
  -H "Content-Type: application/json" \
  -d "{\"fileUrl\": \"$FILE_URL\", \"userId\": \"$USER_ID\", \"fileId\": \"$FILE_ID\"}")

echo "Response:"
echo "$response"

TASK_ID=$(echo "$response" | grep -o '"task_id":"[^"]*' | grep -o '[^"]*$')
echo "Task ID: $TASK_ID"

echo "Polling /admin/jobs/$USER_ID..."
for i in {1..30}; do
  jobs=$(curl -s http://127.0.0.1:8000/admin/jobs/$USER_ID)
  status=$(echo "$jobs" | grep -o "\"file_id\": \"$FILE_ID\", \"status\": \"[^\"]*" | grep -o '[^"]*$')
  echo "Poll $i: Status = ${status:-WAITING_FOR_REGISTRY}"
  if [ "$status" = "SUCCESS" ] || [ "$status" = "FAILED" ]; then
    echo "Done!"
    exit 0
  fi
  sleep 5
done
