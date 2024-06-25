-- https://leetcode.com/problems/not-boring-movies/?envType=study-plan-v2&envId=top-sql-50
select * from Cinema
where description != "boring"
and (mod(id, 2)=1) -- odd number
order by rating desc;

-- https://leetcode.com/problems/average-selling-price/?envType=study-plan-v2&envId=top-sql-50
select p.product_id, ifnull(round(sum(units*price)/sum(units),2),0) as average_price 
from Prices p 
left join UnitsSold u
on p.product_id=u.product_id
and u.purchase_date between p.start_date and p.end_date
group by product_id;