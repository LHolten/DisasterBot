# DisasterBot (WIP)

ExampleBot documentation here: https://github.com/RLBot/RLBotPythonExample/blob/master/README.md

RLBot discord here: https://discord.gg/2prRFJ7

RLUtilities here: https://github.com/samuelpmish/RLUtilities

## Guidelines
_The guidelines are WIP._

### Contribution guidelines

* Documentation can be changed on the master branch.
* All other files should never be modified on the master branch in order to keep the bot stable.
* Use the issues feature in github as a todo list (and for issues).


### Teamwork guidelines

* Don't work on someone else's code without permission, because your pull-request may not be merged.
* Issues can be worked on by everyone, but don't doublecommit.
  * To make sure something is not being worked on, ask in discord.
  * When you start with something, tell us in discord or assign yourself to the issue.
* Use a pull-request to ask others for feedback on your work (feedback can be given on github).
  * After testing changes and approval from the other collaborators a branch can be merged into master.


### Discussion guidelines

* Discuss small decisions on discord via dm.
* Discuss important decisions in the RLBot server.
  * You can also ask for help in this server.
  * Please use the correct channel, #strategy-discussion for strategy discussion etc.


### Programming goals

* The bot should be completely modular e.g. every state, controller, planner should have it's own file and use inheritance.
* Python code should be according to [PEP 8](https://www.python.org/dev/peps/pep-0008/).


## Working plan

* Build different actions that do a specific thing.
* Build a testing environment (using state setting) to test these actions.
* When we are happy with an action it can be used by the policy.
* Build a policy (this part of the plan needs work).


### Project structure
_The project structure is WIP._

* One configuration file to run the bot. 
* One "mechanic" folder that contains the mechanics. (see the decription below)
* One "action" folder that contains the actions.
* One "policy" folder that contains the algorithm that choses between the states.
* One "utils" folder that contains tools used by the bot.

#### Mechanic & Action folders
* One folder for each mechanic/action that contains:
  * One python file that contains the action.
  * One python file that contains a test agent that uses the action.
  * One python file that contains a rlbottraining exercise playlist.
  

## Definition of concepts

### Policy
The policy is responsible for making decisions about what
to do in a specific scenario. It considers what is currently
going on in the game, and selects actions for the bot that
carry out a given strategy.

### Actions
Actions are high-level descriptions of things that a Rocket League
agent can do. Their name implies what they try to do, for example:
* Drive to own goal
* Shoot ball toward opponent goal
* Dribble ball away from the opponent
* Pass ball to teammate

In general, actions are a collection of mechanics working together to accomplish a task.
These actions should have no parameters.
Actions should not contain too much strategy (when in doubt ask).

### Mechanics
Mechanics are the simple building blocks of actions. They try to satisfy
the requirements of the action, and return the actual controls that will be
passed to the bot. For example the actions above may decomposed into the following mechanics:
* Drive to own goal
  * Drive
  * Dodge
  * Wavedash
* Shoot ball toward opponent goal
  * Drive
  * Dodge
  * Aerial
  * Aerial Recovery
* Dribble ball away from the opponent
  * Catch Bounce
  * Drive
  * Dribble
* Pass ball to teammate
  * Drive
  * Dodge
  * Aerial
  * Aerial Recovery
  
Mechanics should have no strategy at all.
RLUtilities are in this category.

