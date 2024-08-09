## Arrays & Hashing

1. Contains Duplicate:

   - Approach: Use a hash set to tack seen elements; if an element is already in
     the set, return True
   - Implementation: Iterate through the array, add each element to a set, and check for duplicates during insertion.
2. Valid Anagram:
    - Approach: Sort both strings and compare, or use a hash map to count character frequencies and compare.
    - Implementation: Sort both strings and compare them, or count the occurrences of each character using a hash map and compare the counts.
3. Two Sum:
    - Approach: Use a hash map to store the indices of elements, and check if the complement of the current element exists in the map
    - Implementation: Iterate through the array, for each element, check if its complement (target - element) is in the hash map, and return the indices if found.
4. Group Anagrams:
    - Approach: Use a hash map where the key is a sorted version of the string and the value is a list of anagrams.
    - Implementation: Sort each string and use the sorted string as a key in the hash map to group anagrams together.
5. Top K Frequent Elements:
    - Approach: Use a hashmap to count frequencies and then a heap (or bucket sort) to find the top k elements.
    - Implementation: count frequencies using a hash map, and then use a heap to extract the K elements with the highest frequencies.


