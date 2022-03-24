--------------------------------------------------------------------------------------------------------------------------
-- STEP 1: Data Quality Check 
--------------------------------------------------------------------------------------------------------------------------
-- Check duplicate: expect NULLs
select USER_ID, 
       count(1) as user_cnt
from `gentle-complex-312720.chime.member_disputes` 
group by 1
having count(1) > 1
;
-- Check missing USER_ID: expect NULLs 
select *
from `gentle-complex-312720.chime.member_disputes` 
where USER_ID is null
;
-- Ensure RISK_SCORE within a range between 0 and 1 
-- 7 observations seem to have invalid risk score 
select *
from `gentle-complex-312720.chime.member_disputes` 
where RISK_SCORE not between 0 and 1
;
-- Ensure both DD and Non-DD members have enrolled flag
select ENROLLED, TYPE, count(1)
from `gentle-complex-312720.chime.member_disputes` 
group by 1,2
-- For non-enrolled users, 28% of them have more than 1 disputes on an account  
select round(count(case when NUMBER_OF_DISPUTES <> 0 then USER_ID end)* 100/ count(USER_ID)) as dispute_pctg
from `gentle-complex-312720.chime.member_disputes` 
where ENROLLED = 0

-- check the number of users who haven't enrolled but have more than 1 dispute on the account
select round(count(case when NUMBER_OF_DISPUTES <> 0 then USER_ID end)* 100/ count(USER_ID)) as dispute_pctg,
       round(count(case when REVENUE <> 0 then USER_ID end)* 100/ count(USER_ID)) as dispute_pctg,
       count(case when NUMBER_OF_DISPUTES <> 0 then USER_ID end) as dispute_cnt,
       count(user_id) as ttl_cnt
from `gentle-complex-312720.chime.member_disputes` 
where ENROLLED = 0
--------------------------------------------------------------------------------------------------------------------------
-- STEP 2: Create a new table that involves total financial cost and net profit 
-- Financial Total Cost = # of disputes * 10 + acquisition costs (varies depending on channel) 
-- Net profit revenue - Financial Total Cost
--------------------------------------------------------------------------------------------------------------------------
drop table if exists chime.member_disputes_cleaned; 
create table chime.member_disputes_cleaned as
with temp_cost as (
select USER_ID,
       CHANNEL, 
       ENROLLED,
       REVENUE,
       NUMBER_OF_DISPUTES,
       RISK_SCORE,
       ifnull(TYPE, 'Not-Enrolled') as TYPE,
       case when CHANNEL ='SEARCH' then 1.00 + NUMBER_OF_DISPUTES * 10
            when CHANNEL ='SOCIAL' then 1.25 + NUMBER_OF_DISPUTES * 10
            when CHANNEL ='ORGANIC' then 0.5 + NUMBER_OF_DISPUTES * 10
            when CHANNEL ='PARTNER' then 1.5 + NUMBER_OF_DISPUTES * 10
            else NUMBER_OF_DISPUTES * 10 end as TOTAL_COST
from `gentle-complex-312720.chime.member_disputes` 
where RISK_SCORE between 0 and 1 -- remove users with invalid risk scores
)
select *, 
       REVENUE - TOTAL_COST as NET_PROFIT
from temp_cost
;



