# Engineering Manager's Log

> One page. This is where you show us how you *directed* the AI — it matters as much
> as the code. Be concrete. Bullet points are fine.

**Name:** Andrew Landis
**Time spent (be honest):** 1hr 15min

---

## How I broke the work down
<!-- The plan you gave the AI. Tasks, order, what you did first and why. -->
I worked with opencode using DeepSeek V4 Pro. The first thing I had to figure out was my client and human-in-the-loop gate so I got my agent up to speed on the project by having it review the project and provide suggestions, ending up on a CLI.

Next was figuring out the workflow I wanted to implement and gating the write client. Given the endpoints in the API, getting all emails at once and processing through triage, approval, action made the most sense. I had the agent implement features one step at at time. Here's a step by step and what happened
1. implement api functions
1. create cli client and read-only triage client
1. retrieve inbox
1. call triage_inbox with llm
- switched here from ollama cloud kimik2 to claude haiku due to unknown ollama 403
- removed a write_client that the initial implementation tried to pass in at this step
1. iterated on user workflow with info formatting batching actions
- added a summary page to show what was going to happen
- clearing terminal for clear communication to user on what the current email/approval was
1. Ad hoc testing

I did run into one decision that I would normally review given time, which was that "unknown" classifications from the AI analsysis were defaulting to spam and dropped. I would suggest another branch of features for a action review process that the user would be able to choose from the actions for an email.

Once ad hoc testing passed and it looked like I had a working project, I moved on to testing. As I was cutting it close on time I had to let the AI rip here and pushed forward with some of the AI testing principles I have recently been looking into, with example, property, and mutation testing. Time ran out with me getting mutation testing running.


## Where I ran things in parallel
<!-- Multiple threads / agents / tabs? What ran at once, and how you kept them straight. -->
As I'm writing this! The mutation testing suite was getting up and running and I have a result of the following: Mutation score: 62.5% (257 killed / 411 total, 154 survived). So there's plenty of work to do on the test suite and reviewing/validating there. Otherwise I had tmux panes up of separate opencode instances to implement the backend functionality while getting my first feature set up for the client.

## One time the AI was wrong, and how I caught it
<!-- The single most useful thing to tell us. What did it get wrong, how did you notice,
     and what did you do about it? -->
It tried to introduce the write client and the execute function into the triage_inbox function which would have failed the requirement for not having the write client in memory while spam was present.

## What I deliberately cut to fit the 2 hours
<!-- What you chose NOT to build, and the tradeoff you accepted. -->
Evaluating the test suite more thorougly and being able to analyze the mutation testing results for providing a path foward there.

## The design decision I'm proudest of
<!-- Especially around the human-in-the-loop gate or least-privilege handling. -->
This implementaiton allows you to approve per-action not just per email so you are able to draft a reply and not create a CRM lead for the sales_lead if desired. It could come across cleaner in the CLI but I liked that option.

Finding the spam security issue and also just this project in general. Admittedly I'm not the most comfortable with having my hands this much off the wheel when it comes to letting AI generate the majority of project code without my review, but this was a good challenge and push for me to get out of my comfort zone which I need to do to stay up to date with the capabilities of the tooling we now have.
