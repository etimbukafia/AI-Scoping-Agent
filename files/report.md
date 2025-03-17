# DOCUMENTED BEHAVIOR

## Stated Contract Purpose and Functionality

- The contract is designed to provide an oracle solution for fetching `scrvUSD` vault parameters from Ethereum and providing them on other chains.
- The goal is to compute the growth rate in a safe and precise manner, ensuring no losses due to approximation.
- The oracle supports creating stableswap-ng pools for other assets like `USDC/scrvUSD`, `FRAX/scrvUSD`, etc.

> "This project contains a solution that fetches scrvUSD vault parameters from Ethereum, and provides them on other chains, with the goal of being able to compute the growth rate in a safe (non-manipulable) and precise (no losses due to approximation) way."

## Described Business Logic and Security Model

- The oracle fetches `scrvUSD` vault parameters from Ethereum and provides them on other chains.
- The implementation includes different versions (`v0`, `v1`, `v2`) with varying assumptions about the behavior of `scrvUSD`.
- The oracle uses a blockhash feed solution to ensure the correctness of the parameters.
- The oracle is controlled by a DAO, allowing parameters to be changed by vote.

> "Hence, the initial version named `v0` simply replicates scrvUSD and returns the real price at some timestamp. Considering the price is always growing, `v0` can be used as a lower bound. `v1` introduces the assumption that no one interacts with scrvUSD further on. This means that rewards are being distributed as is, stopping at the end of the period. And no new deposits/withdrawals alter the rate. This already gives a great assumption to the price over the distribution period, because the rate is more or less stable over a 1-week period and small declines do not matter. `v2` adds an assumption that rewards denominated in crvUSD are equal across subsequent periods."

## Documented Component Interactions

- The oracle interacts with the `scrvUSD` vault on Ethereum to fetch parameters.
- The oracle provides these parameters to other chains to compute the growth rate.
- The oracle uses a blockhash feed solution for parameter updates.
- The oracle supports the creation of stableswap-ng pools for various assets.

## Specified User Flows and Permissions

- Users can interact with the oracle to fetch `scrvUSD` vault parameters.
- The DAO controls the oracle and can change its parameters through voting.

## Defined System Constraints

- The oracle should not jump more than 1 bps per block.
- The safe threshold for the oracle is 0.5 bps.
- The upper bound of price growth is considered to be 60% APR.

> "Taking the minimum pool fee as 1bps means that the oracle should not jump more than 1 bps per block. And for extra safety, take 0.5bps as a safe threshold. Therefore, we consider that scrvUSD will never be over 60% APR."

# SECURITY SPECIFICATIONS

## Documented Security Features

- The oracle uses a blockhash feed solution to ensure the correctness of the parameters.
- The oracle includes smoothing to handle sudden updates safely.
- The oracle is controlled by a DAO, allowing parameters to be changed by vote.

## Stated Access Control Rules

- The DAO controls the oracle and can change its parameters through voting.

## Described Risk Mitigations

- The oracle uses a blockhash feed solution to ensure the correctness of the parameters.
- The oracle includes smoothing to handle sudden updates safely.
- The oracle is controlled by a DAO, allowing parameters to be changed by vote.

## Emergency Procedures

- No emergency procedures are documented.

## Known Limitations

- The oracle can rarely provide an incorrect blockhash, but not an incorrect block number.
- The upper bound of price growth is considered to be 60% APR.

> "It can __rarely__ provide an incorrect blockhash, but not an incorrect block number. Thus, a new update with a fresh block number will correct the parameters."

# SYSTEM ARCHITECTURE

## Documented System Components

- The oracle fetches `scrvUSD` vault parameters from Ethereum.
- The oracle provides these parameters to other chains.
- The oracle uses a blockhash feed solution for parameter updates.
- The oracle supports the creation of stableswap-ng pools for various assets.

## Described Integration Points

- The oracle integrates with the `scrvUSD` vault on Ethereum.
- The oracle integrates with other chains to provide parameters.
- The oracle integrates with a blockhash feed solution for parameter updates.
- The oracle integrates with stableswap-ng pools for various assets.

## Stated Economic Model

- The economic model is not explicitly stated.

## Listed Deployment Parameters

- The deployment parameters are not explicitly listed.

## Configuration Requirements

- The oracle is controlled by a DAO, allowing parameters to be changed by vote.

# IMPLICIT ELEMENTS

## Unstated Assumptions

- The oracle assumes that the `scrvUSD` price is always growing.
- The oracle assumes that the rate is more or less stable over a 1-week period.
- The oracle assumes that rewards denominated in `crvUSD` are equal across subsequent periods.

## Unclear Specifications

- The documentation does not clearly specify the economic model of the oracle.
- The documentation does not clearly specify the deployment parameters of the oracle.

## Ambiguous Behaviors

- The behavior of the oracle in case of an incorrect blockhash is not clearly defined.
- The behavior of the oracle in case of a market crash is not clearly defined.

## Documentation Gaps

- The documentation does not provide emergency procedures.
- The documentation does not provide a clear economic model.
- The documentation does not provide clear deployment parameters.
- The documentation does not provide a clear description of the user flows and permissions for interacting with the oracle.
- The documentation does not provide a clear description of the system constraints for the oracle.