-- https://leetcode.com/problems/combine-two-tables/description/
/*
Approach 1: Using outer join
Intuition
Since the PersonId in table Address is the foreign key of table Person, we can join these two tables to get the address information of a person.

Considering there might be no address information for every person, we should use outer join instead of the default inner join.

*/
SELECT firstName, lastName, city, state from Person
left join Address
on Person.personId=Address.personId;